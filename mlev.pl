#!/usr/bin/perl

use strict;
use warnings;

use Cwd;
use File::Copy;

use File::Basename;
use lib dirname (__FILE__);
use Common;
use TSV;

use lib '.';

my $mlev_dir = dirname (__FILE__);
my $cwd = getcwd;

use MLEV;
use SMLModel;
use Fcntl ':flock'; # Import LOCK_* constants
use ParallelRunner;
use FileGarbageBin;

my $tmpdir = MLEV::get_mlev_tmpdir(); # "/tmp";


my $usage = "
$0: A pipeline for evaluating machine learning models

Usage:
 * Training:
   ./mlev.pl  train    model_output  training_data  model_spec 
     
 * Testing:
   ./mlev.pl  test         model_input   test_data
   ./mlev.pl  test-thres   model_input   test_data # including threshold results

 * Predict:
   ./mlev.pl  predict  model_input   test_data
   
 * Training and test:
   ./mlev.pl  train-test  training_data  test_data  model_spec

 * Feature select:
   ./mlev.pl  select  [dataset|-] /features_to_include/ ... /-features_to_exclude/   

 * Split data:
   ./mlev.pl  split  fraction1,fraction2,[xrepeats] 
   
 * Split data (cross-validation)
   ./mlev.pl  cvsplit  training_data  folds[xrepeats]
   
 * Random (training-test) split evaluation
   ./mlev.pl  tttest  training_data  fraction1[xrepeats]  [PREP:feature_prep_pipeline_spec] ... [model_spec] ...
   
 * Cross-validation evaluation
   ./mlev.pl  cvtest  training_data  folds[xrepeats]  [PREP:feature_prep_pipeline_spec] ... [model_spec] ...
   
 * General resampling function
   ./mlev.pl  rstest  split_data_spec [PREP:feature_prep_pipeline_spec] ... [model_spec] ...
   
 * General resampling function - cleaning
   ./mlev.pl  rstest-clean  split_data_spec
   
";

my $action = shift @ARGV;

MAIN: {
	my @output = ();
	for ($action) {
		! defined and do { die $usage; last; };
		/^train$/      and do { @output = &do_train(@ARGV);   push @output, "\n"; last; };
		/^test$/       and do { @output = &do_test(@ARGV);    last; };
		/^test-thres$/ and do { @output = &do_test('--threshold', @ARGV);    last; };
		/^train-test$/ and do { @output = &do_train_test(@ARGV); last; };
		/^select$/     and do { @output = &do_select(@ARGV);     last; }; 
		/^split$/      and do { @output = &do_split(@ARGV);      last; };
		/^cvsplit$/    and do { @output = &do_cvsplit(@ARGV);    last; };
		/^cvtest$/     and do { @output = &do_cvtest(@ARGV);     last; };
		/^tttest$/     and do { @output = &do_tttest(@ARGV);     last; };
		/^rstest$/     and do { @output = &do_rstest(@ARGV);     last; };
		/^rstest-clean$/     and do { @output = &do_rstest_clean(@ARGV);     last; };
		/^predict$/    and do { @output = &do_predict(@ARGV[0,1]); last; };
		/^toarff/      and do { @output = &do_toarff(@ARGV); last; };
		die "Invalid action ''$action'' specified.\n$usage";
	};
	print @output;
# 	<STDIN>;
	exit;
}

########################################################################################
sub run {
	my $cmd = shift;
	mlev_debug(MLEV::DL_INVOKE, "> $cmd");
# 	print STDERR "\e[1;37m> $cmd\e[0m\n";
	return system $cmd;
}

sub get_fn_datastem {
	my $fn_data = shift;
	my $fndatastem = $fn_data;
	$fndatastem =~ s|.*/||;
	$fndatastem =~ s|.tsv$||;
	$fndatastem = '' if $fndatastem eq '-';
	return $fndatastem ;
}


########################################################################################
sub run_pipeline($\@;\@) {
	my $fn_data         =    $_[0];
	my @pipeline        = @{ $_[1] };
	my $OUT_last_input_lines =    $_[2] ;  # A variable to store input lines

	my @wanted_outputs;
	my $fn_last_input ;
	
	for my $p (@pipeline) {
		while ( $p =~ /\{DATA(\d+)\}/ ) {
			my $repl = $&;
			my $serno = $1;
			my $tmpfile = mlev_tmpfile( "pl-S".$serno );
			if ( $serno == 0 ) {
				$p =~ s/\Q$repl\E/$fn_data/;
			} else {
				$p =~ s/\Q$repl\E/$tmpfile/;
			}
		}
		
		if ($p =~ /META:\s*(.+?)\t(.*)/) {
			my ($a, $v) = ($1, $2);
			my @vparts = split /\t/, $v;
			@wanted_outputs = @vparts if ( $a =~ /OUTPUT/ );
			$fn_last_input = $v if ( $a =~ /LAST_INPUT/ );
			next;
		}
		
#  		print STDERR "\e[1;35m! $p\e[0m\n";
 		run $p;
	}

	my @retval = map { file($_) } @wanted_outputs;
	
	@{ $OUT_last_input_lines } = file($fn_last_input) if ( defined($OUT_last_input_lines) ) and (defined $fn_last_input) ;
	
	return @retval ;
}

########################################################################################
sub prepare_pipeline_find_cmd($) { # try to find the command to execute the pipeline element
	my $cmd = shift;
	for my $dir ($cwd, $mlev_dir, ".") {
		for my $cmdpart ( "$cmd", "$cmd.pl", "script/$cmd", "script/$cmd.pl", "bin/$cmd", "bin/$cmd.pl", "$cmd" , "$cmd.pl" ) {
			for my $postfix ( "", ".pl", ".py" ) {
				my $cmdfile = join("/", $dir, $cmdpart.$postfix );
				return $cmdfile if ( -f $cmdfile and -x $cmdfile ) ;
			}
		}
	}
	warn "$0: WARNING: Pipeline element executable ''$cmd'' not found (cwd = $cwd). ";
	return $cmd ;
}

########################################################################################
sub prepare_pipeline_transform_cmd($$$$$;$) { 
	my $pipeline_action = shift;
	my $fn_model  = shift;
	my $df_no_in  = shift;
	my $df_no_out_ref = shift;
	my $raw_pipeline = shift;
	my $params = shift;
	my ($cmd, @args) = split /\s+/, $raw_pipeline;
	
	++ $$df_no_out_ref;
	
	my $fn_output = "{DATA$$df_no_out_ref}";
	my $fn_input = "{DATA$df_no_in}";
	if ( $cmd !~ /^CL-/ ) {
		unshift @args, $fn_output  ;
	} else {
		push @args, "> $fn_output" ;
	}
	
	my $final_cmd = '';
	if ( defined $params) {
		if ( exists $$params{'clone_model'} ) {
			++ $$df_no_out_ref;
			$final_cmd = "/usr/bin/cp -v $fn_model {DATA$$df_no_out_ref}; " ;
			$fn_model = "{DATA$$df_no_out_ref}";
		}
	}
		
	unshift @args, $fn_input ;
	unshift @args, $fn_model ;
	unshift @args, $pipeline_action ; 
	$final_cmd .= join(" ", prepare_pipeline_find_cmd($cmd), @args);
	
#   print STDERR "\e[1;33m".join("|", "[$f_model_to_predict]", $cmd, @rest)."\e[0m\n";
#   print STDERR "\e[1;35m$final_cmd\e[0m\n";

	return ($final_cmd, $fn_output, $fn_input);
}


########################################################################################
sub prepare_pipeline($$\@$) { # prepare the pipeline before running.
    my $pipeline_action  =    $_[0];
    my $data_file        =    $_[1];
    my @raw_pipeline     = @{ $_[2] };
    my $fn_model         =    $_[3]; # model_name
    
    my @prepared_pipeline;
	
	my $df_serno = 0;
	my $df_serno_out = 0;
	
	my @fn_outputs;
	my $fn_last_input;
    for my $pi ( 0 .. $#raw_pipeline ) {
		my $p = $raw_pipeline[$pi];
		if ( $p =~ /^\{(.+)\}$/ ) {
			$p = trim($1);
			my @subparts = split /\s*;\s*/, $p ;
			my $df_serno_out = $df_serno + 1;
			for my $sp (@subparts) {
				my ($final_cmd, $fn_output, $fn_input) = prepare_pipeline_transform_cmd($pipeline_action, $fn_model, $df_serno, \$df_serno_out, $sp, {clone_model=>1} ) ; 
				push @prepared_pipeline, $final_cmd;
				push @fn_outputs, $fn_output if $pi == $#raw_pipeline;
				$fn_last_input = $fn_input if $pi == $#raw_pipeline;
			}
			$df_serno = $df_serno_out ;
		} else {
			my ($final_cmd, $fn_output, $fn_input) = prepare_pipeline_transform_cmd($pipeline_action, $fn_model, $df_serno, \$df_serno_out, $p, {is_last_command=>($pi == $#raw_pipeline)});
			push @prepared_pipeline, $final_cmd;
			push @fn_outputs, $fn_output if $pi == $#raw_pipeline;
			$fn_last_input = $fn_input if $pi == $#raw_pipeline;
		}
		$df_serno = $df_serno_out ;
	}
	push @prepared_pipeline, join("\t", "META: OUTPUT", @fn_outputs);
	push @prepared_pipeline, join("\t", "META: LAST_INPUT", $fn_last_input);
	mlev_debug( MLEV::DL_PIPELINE,  map { "- $_" } @raw_pipeline);
# 	print STDERR map { "\e[1;33m- $_\e[0m\n" } @raw_pipeline;
# 	print STDERR map { "\e[1;33m+ $_\e[0m\n" } @prepared_pipeline;
    return @prepared_pipeline;
}


########################################################################################
sub arg_to_pipeline($) {  # Converting commandline arguments to pipeline
	my $arg = shift;
	my @pipeline = map { trim($_) } split /\s*\|\s*/, $arg ;
    return @pipeline;
}

######################################################################################
sub register_model($$\@) {
    my $fn_output_model = $_[0];
    my $fn_training_data = $_[1];
    my @pipeline = @{ $_[2] } ;
    
	my $model = SMLModel->new($fn_output_model);
	$model->set_attr('training_data', $fn_training_data);
	if ( $model->has_attr('pipeline') ) {
		my @prep_pipeline = $model->get_attr('pipeline');
		if ( $prep_pipeline[$#prep_pipeline] ne $pipeline[$#pipeline] ) {
			@pipeline = (@prep_pipeline, @pipeline); 
			mlev_debug( MLEV::DL_INFO,  "Merging pipelines");
		}
	}
	
	$model->set_attr('pipeline', @pipeline);
	$model->save();
	return $model ;
}

######################################################################################
sub do_train {  # Main subroutine: model training
	my @args = @_;
	
	my $fn_output_model = shift @args;
	
	my $fn_training_data = shift @args;
	
	my $model_spec = shift @args;

	mlev_debug( MLEV::DL_INFO, "do_train(".join(", ", $fn_output_model, $fn_training_data, $model_spec, @args).")" );

	die "$0: FATAL: Invalid model specification.\n$usage" if ! defined $model_spec;
	
	my @pipeline = arg_to_pipeline($model_spec);
	
# 	unlink $fn_output_model if -f $fn_output_model ; # this is fresh training. remove this model if already exists
	
	my $model = register_model($fn_output_model, $fn_training_data, @pipeline);
	
	my @prepared_pipeline = prepare_pipeline('train', $fn_training_data, @pipeline, $fn_output_model);
	
    return run_pipeline($fn_training_data, @prepared_pipeline); 
}


######################################################################################
sub do_predict { # Main subroutine: model prediction
	my @args = @_;
	
	my $fn_input_model = shift @args;
	my $fn_test_data = shift @args;
	my $fn_last_tsv_input_ref = shift @args;
	
	die "$0: FATAL: Model file not supplied.\n$usage" if ! defined $fn_input_model ;
	die "$0: FATAL: Model file ($fn_input_model) not readable.\n$usage" if ! -f $fn_input_model ;
	
	die "$0: FATAL: Test data not supplied.\n$usage" if ! defined $fn_test_data ;
	die "$0: FATAL: Test data ($fn_test_data) not readable.\n$usage" if ! -f $fn_test_data ;
	
	my $model = SMLModel->new($fn_input_model);
	
	my @pipeline = $model->get_attr('pipeline');
	
	my @prepared_pipeline = prepare_pipeline('predict', $fn_test_data, @pipeline, $fn_input_model);  # Preparing the pipeline for testing
	
	my @last_input_lines;
	
	my @retval = run_pipeline($fn_test_data, @prepared_pipeline, @last_input_lines);   # Should return "Class1\tClass2\nProbability1\tProbability2\n"  
	
# 	print STDERR $fn_last_tsv_input_ref ;
	
	if ( defined $fn_last_tsv_input_ref ) {
		@{ $fn_last_tsv_input_ref } = @last_input_lines;
	}
	
	return @retval;
}


######################################################################################
sub do_test { # Main subroutine: model testing
	my @args = @_;
	my $f_threshold = 0;
	if ( $args[0] eq '--threshold' ) {
		$f_threshold = 1 ;
		shift @args;
	}
	my $fn_input_model = shift @args;
	my $fn_test_data = shift @args;
	
	my @last_input_tsv_lines ;
	my @output = do_predict($fn_input_model, $fn_test_data, \@last_input_tsv_lines);
	
	chomp for @output;
	
    my $TSV_test = TSV->new; $TSV_test->import_data( @last_input_tsv_lines );
    my $class_name = $TSV_test->guess_class_label;
    my @correct_labels = map { ${$_}{$class_name} } (@{ $TSV_test->{'data'} } ) ;
	
# 	print STDERR "fn_input_model: $fn_input_model\n";
# 	print STDERR "fn_test_data: $fn_test_data\n";
# 	print STDERR map { "LABELS\t[$_]\n" } @correct_labels;
# 	print STDERR map { "OUTPUT\t$_\n" } @output;
	
	my @rows = @output;
	my $header = shift @rows;
	my @class_labels = map { trim($_) } split /\t/, $header;
# 	print STDERR map { "[$_]\n" } @class_labels;

	my @data;
	for my $row (@rows) {
#         print $row;
		my @values = split /\t/, $row;
		my %a = map { $class_labels[$_] => trim($values[$_]) } (0 .. $#class_labels);
		push @data, \%a;
	}

    my @confmat;
    for my $i ( 0 .. $#class_labels) {
        for my $j ( 0 .. $#class_labels) {
            $confmat[$i][$j] = 0;
        }
    }

    my $N_correct = 0;
    my $N_tested  = 0;
    for my $r (0..$#data) {
        next if $correct_labels[$r] =~ /^(?:NA|\?|\s*)$/;
        my @a = map { $data[$r]{$_} } @class_labels;
        my $max_at = 0; 
        my ($label_at) = grep { $class_labels[$_] eq $correct_labels[$r] } (0 .. $#class_labels);
        die "No matching label $correct_labels[$r]." if ! defined $label_at or ! defined $correct_labels[$label_at];
        for my $i (1..$#a) { 
            next if $a[$max_at]=~ /^(?:NA|\?|\s*)$/;;
            $max_at = $i if ( $a[$i] > $a[$max_at] ); 
            }
        $confmat[$label_at][$max_at] ++ if defined $label_at;
        $N_tested ++;
        $N_correct ++ if defined $label_at and $label_at == $max_at;
    }

    my $accuracy = ( $N_correct / $N_tested );
    
    my @Loutput;
    
    push @Loutput, "Accuracy\t$accuracy\n";
    push @Loutput, "ConMat\t\t".join("\t", @class_labels)."\n";
    for my $i ( 0 .. $#class_labels) {
        push @Loutput, join("\t", "ConMat", $class_labels[$i], ( map { $confmat[$i][$_] } ( 0 .. $#class_labels )) )."\n";
    }
    
	for my $l (@class_labels) {
		my @x ;
		my @y; 
		for my $i ( 0 .. $#data) { # next if $correct_labels[$i] =~ /^(?:NA|\?|\s*)$/;
			my $s = $data[$i]{$l};
			if ( $correct_labels[$i] eq $l ) {
				push @x, $s;
			} else {
				push @y, $s;
			}
		}
		
		if ($f_threshold) {
			my @auroc_thres ;
			my $auroc2 = auroc(@x, @y, @auroc_thres);
			my @thres_labels = qw(t tp fp fn tn sens spec ppv npv eauc f1);
			my $best_row = undef;
			my $max_eauc = -1;
			push @Loutput, "AUC\t$l\t$auroc2\n";
			push @Loutput, join("\t", "Threshold.Label", $l, @thres_labels)."\n";
			for my $row (@auroc_thres) {
				if ( $$row{eauc} > $max_eauc ) {
					$max_eauc = $$row{eauc} ;
					$best_row = $row;
				}
				push @Loutput, join("\t", "Threshold", $l, (map { $$row{$_} } @thres_labels)   )."\n";
			}
			push @Loutput, join("\t", "Threshold.Optimal", $l, (map { $$best_row{$_} } @thres_labels)   )."\n" if defined $best_row;
		} else {
			my $auroc2 = auroc(@x, @y);
			push @Loutput, "AUC\t$l\t$auroc2\n";
		}
	}
	
    # Calculate average by Hand and Till method:
    
    my @valid_AUCs;
	for my $l1i ( 0.. $#class_labels ) {
        my $l1 = $class_labels[$l1i];
        for my $l2i ( ($l1i+1) .. $#class_labels ) {
            my $l2 = $class_labels[$l2i];
            my ( @x, @y );
            for my $i ( 0 .. $#data) { 
                my $cl = $correct_labels[$i] ;
                next if ( $cl ne $l1 ) and ( $cl ne $l2 ) ;
                my $s = $data[$i]{$l1};
                if ( $cl eq $l1 ) { push @x, $s; } else { push @y, $s; }
			}
            next if ! ( scalar(@x) + scalar(@y) );
# 			my @auroc_thres ;
#             my $auroc2 = auroc(@x, @y, @auroc_thres);
#             push @valid_AUCs, $auroc2 ;
# 			push @Loutput, "AUC-pw\t$l1:$l2\t$auroc2\n";
# 			for my $row (@auroc_thres) {
# 				push @Loutput, join("\t", "Thres", "$l1:$l2", $$row{t}, $$row{tp} , $$row{fp} , $$row{fn} , $$row{tn}, $$row{sens}, $$row{spec} )."\n";
# 			}
			my $auroc2 = auroc(@x, @y);
            push @valid_AUCs, $auroc2 ;
			push @Loutput, "AUC-pw\t$l1:$l2\t$auroc2\n";
		} 
	}

	my $mean_auroc = mean(@valid_AUCs);
    push @Loutput, "AUC\tAverage\t$mean_auroc\n";
	

    return (@Loutput) ;
}


######################################################################################

our $do_train_test_serno ;

$do_train_test_serno = 1;

sub do_train_test {
	
	my $fn_save_model = undef;
	
	my @args;
	for my $arg (@_) {
		if ( $arg =~ /^--keep-model=(.+)/ ) {
			$fn_save_model = $1;
			next;
		}
		push @args, $arg ;
	}
	
	my $fn_train = shift @args ; 
	my $fn_test = shift @args ; 
	my $model_spec = shift @args ; 
	my $part_trained_model_fn = shift @args ; # partially trained model
	
# 	print STDERR "fn_train = $fn_train.
# 	print STDERR "fn_test = $fn_test\n";
# 	print STDERR "model_spec = $model_spec\n";
	
	my $seq = sprintf("%04d", $do_train_test_serno);
	my $fn_model = mlev_tmpfile("tr-ts-$seq.model");
	
    mlev_debug( MLEV::DL_INFO, "do_train_test(".join(", ", ( map { $_ // '' } ($fn_train, $fn_test, $model_spec, $part_trained_model_fn) ))." -> $fn_model" );
	
	# Make a copy of partially trained model 
	mlev_debug( MLEV::DL_INVOKE, "> cp $part_trained_model_fn $fn_model" ) if ( defined $part_trained_model_fn ) ;
	
	copy($part_trained_model_fn, $fn_model) if ( defined $part_trained_model_fn ) and ( -f $part_trained_model_fn );
	
	do_train($fn_model, $fn_train, $model_spec);
	
	my @Routput = do_test($fn_model, $fn_test);
	
	if ( (defined $fn_model) and (-f $fn_model) ) {
		if ( defined $fn_save_model ) {
			mlev_debug( MLEV::DL_INFO, "do_train_test(): saving $fn_model to $fn_save_model " );
			copy($fn_model, $fn_save_model) 
		}
		unlink $fn_model ;
	}
	return @Routput;
}

######################################################################################
sub load_primary_tsv {
    my $fn_data = shift;
	my $tsv;
	
	if ( $fn_data eq '-' ) {
        $tsv = TSV->new;
        my @lines = <STDIN>;
        $tsv->import_data( @lines );
    } else {
        $tsv = TSV->new($fn_data);
    }
    return $tsv;
}

######################################################################################
sub do_select { # Main subroutine: feature selection
	my @args = @_;
	my $fn_data = shift @args;
	
	my $tsv = load_primary_tsv($fn_data) ;
	
	my @fields = @{ $tsv->{'fields'} };
	
	my @wanted_fields = @fields;
	
	for my $a (@args) {
		my $regex = $a;
		if ( $regex =~ /^\-(.+)/ ) {
			$regex = $1; 
			@wanted_fields = grep { ! /$regex/ } @wanted_fields ;
		}
        elsif ( $regex =~ /^\+(.+)/ ) {
			$regex = $1;
			my @fields_to_add = grep { /$regex/ } @fields ;
			@wanted_fields = (@wanted_fields, @fields_to_add);
		}
		else {        
			@wanted_fields = grep { /$regex/ } @wanted_fields ;
		}
	}
	
    my $tsv1 = $tsv->select(@wanted_fields);
    return $tsv1->to_string();
}


######################################################################################
sub sum { my $x = 0; while (scalar (@_)) { $x += shift; } return $x; }

######################################################################################
sub do_split { # Main subroutine: spliting data
	my @args = @_;
	my $fn_data = shift @args; # data set
	my $fracs   = shift @args; # fractions
	my $repeats = undef;       # repeats
	$repeats = $1 if $fracs =~ s/x(\d+)$//;
	
	my $tsv = load_primary_tsv($fn_data) ;
	my $nrow = scalar @{ $tsv->{'data'} };
	
	my @fracs = split /,/, $fracs;
	my @ifracs = map { int($nrow*$_+0.5) } @fracs;
	push @ifracs, scalar( @{ $tsv->{'data'} } ) - sum(@ifracs);
	
	my $fndatastem = get_fn_datastem($fn_data);

	my @output ;
	
	for (my $rep=0; $rep<(defined $repeats ? $repeats : 1); ++$rep) {
		my $pid = $$;
		my $repstr = (defined $repeats) ? "-rep".($rep+1) : '';
		my $fmt_tmppath = $tmpdir."/mlev-$fndatastem-$pid$repstr-%s";
		my @paths;
		my @tsvs;
		
		$tsv->shuffle_rows();
		$tsv->stratify_by( $tsv->guess_class_label );
		
		my $cursor = 0;
		for my $i (0 .. $#ifracs) {
# 			printf STDERR join("\t", $rep, $i, $cursor, ($cursor+$ifracs[$i]-1))."\n";
			my $tsv1 = $tsv->subsample( $cursor .. ($cursor+$ifracs[$i]-1) );
			push @tsvs, $tsv1;
			if ( scalar(@ifracs) == 2 ) {
				$paths[$i] = sprintf($fmt_tmppath, ( $i == 0 ) ? "tr.tsv" : "ts.tsv") ;
			} else {
				$paths[$i] = sprintf($fmt_tmppath, sprintf("part%02d", $i+1) ) ;
			}
			$cursor += $ifracs[$i];
		}
		
		$tsvs[$_]->save_as( $paths[$_] ) for (0 .. $#tsvs);
		
		push @output, join( "\t", map { $paths[$_] } (0 .. $#paths) )."\n";
	}
	return @output;
}


######################################################################################
sub do_cvsplit { # Main subroutine: spliting data
	my @args = @_;
	my $fn_data = shift @args;  # data set
	my $folds   = shift @args;  # folds
	my $repeats = undef;        # repeats
	$repeats = $1 if $folds =~ s/x(\d+)$//;
	
	my $tsv = load_primary_tsv($fn_data) ;
	my $nrow = scalar @{ $tsv->{'data'} };
	
	my $split_point = int( ($nrow / $folds) + 0.5 );
		
	my $fndatastem = get_fn_datastem($fn_data);
	
	my @output ;
	for (my $rep=0; $rep<(defined $repeats ? $repeats : 1); ++$rep) {
		my $pid = $$;
		my $repstr = (defined $repeats) ? "-rep".($rep+1) : '';
		my $fmt_tmppath = $tmpdir."/mlev-$fndatastem-$pid$repstr-fold%d-%s";
		my @paths;
		
		$tsv->shuffle_rows();
		$tsv->stratify_by( $tsv->guess_class_label );
		
		for my $fold (1 .. $folds) {
			my @indices_ts;
			my @indices_tr; 
			for my $i (0 .. ($nrow-1) ) {
                my $s = int ( ( ($fold-1) / $folds ) * $nrow );
                my $e = int ( (  $fold    / $folds ) * $nrow );
                if ( $i >= $s && $i < $e) {
                    push @indices_ts, $i ;
                } else {
                    push @indices_tr, $i ;
                }
			}
			
			my $tsv_tr = $tsv->subsample( @indices_tr );
			my $tsv_ts = $tsv->subsample( @indices_ts );
			my $fn_tsv_tr = sprintf($fmt_tmppath, $fold, "tr.tsv") ;
			my $fn_tsv_ts = sprintf($fmt_tmppath, $fold, "ts.tsv") ;
			my $fn_model_fold = sprintf($fmt_tmppath, $fold, "model.model") ;
			$tsv_tr->save_as( $fn_tsv_tr ) ;
			$tsv_ts->save_as( $fn_tsv_ts ) ;
			push @output, join( "\t", $fn_tsv_tr, $fn_tsv_ts, $fn_model_fold)."\n";
		}
	}
	return @output ;
}

###################################################################################
sub arith_mean {
	my @a = grep { defined and $_ ne 'NA' } @_;
	my $n = scalar(@a);
	my $s = 0.0;
	$s += $_ for (@a);
	return 'NA' if $n == 0;
	return $s / $n;
	}


sub se {
	my $total = 0;
	my @x = grep { defined and ($_ ne 'NA') } @_;
	return 'NA' if scalar(@x) <= 1; 
	my $m = arith_mean(@x);
	for my $v (@x) {
		my $e = $v - $m;
		$total += $e * $e ;
		}
	return sqrt ($total / (scalar(@x) - 1) );
	}

	
######################################################################################
sub do_train_test_wrapper($$$$$;$) {   
    my $trfn = $_[0]; 
    my $tsfn = $_[1];
    my $model_fn = $_[2]; # INPUT/OUTPUT binary model file
    my $model_spec = $_[3]; # INPUT model specification
    my $output_temp_fn = $_[4];
    my $results_prefix = $_[5];
    my @output1 = do_train_test($trfn, $tsfn, $model_spec, $model_fn); # model_fn=NULL if new model
    wait;    
    open(my $fh, '>>', $output_temp_fn) or die "Could not open '$output_temp_fn' - $!";
    flock($fh, LOCK_EX) or die "Could not lock '$output_temp_fn' - $!";
    if ( defined $results_prefix ) {
		print $fh map { join("\t", $results_prefix, "$trfn:$tsfn", $_) } @output1;
    } else {
		print $fh map { join("\t", "$trfn:$tsfn", $_) } @output1;
    }
    close($fh) or die "Could not write '$output_temp_fn' - $!";
    wait;
}


######################################################################################
sub do_cvtest_run_train_test($\@;$) {
	my $model_spec = shift ;
	my @tt_sets = @{ $_[0] }; chomp for @tt_sets ;
	
    my $result_file = "$tmpdir/mlev-$$-cvtest-output.txt";
    $do_train_test_serno=1;
    if ( $model_spec =~ /CL-(?:Meta|Inception|TFImage|FastText|DocumentClassifier)/ ) { # If meta classifiers are used, we should run a thread instead of using all CPUs. # 
		for my $ttset (@tt_sets) {
			my ($trfn, $tsfn, $modelfn) = split /\t/, $ttset ;
			do_train_test_wrapper($trfn, $tsfn, $modelfn, $model_spec, $result_file);
			++$do_train_test_serno;
		}
    } else {
		my $prun= ParallelRunner->new; # (1);
# 		print STDERR "PARALLEL: $_\n" for @tt_sets;
# 		sleep 3;
		for my $ttset (@tt_sets) {
			my ($trfn, $tsfn, $modelfn) = split /\t/, $ttset ;
			# my @output1 = do_train_test($trfn, $tsfn, $model_spec);
			# push @output, map { "$trfn:$tsfn\t$_" } @output1;
			$prun->run(\&do_train_test_wrapper, $trfn, $tsfn, $modelfn, $model_spec, $result_file);
			++$do_train_test_serno;
		}
		$prun->wait();
	}
	
	my @retval = file($result_file) ;
	unlink $result_file;
	return @retval;
}


######################################################################################
sub do_tttest_run_train_test(\@$) { # single training test split, multiple models
	my @model_spec = @{ $_[0] } ;
	my $ttset = $_[1]; chomp $ttset ;
	
    my $result_file = "$tmpdir/mlev-$$-cvtest-output.txt";
    $do_train_test_serno=1;
    if ( grep { /CL-(?:Meta|Inception|TFImage|FastText)/ } @model_spec ) { # If meta classifiers are used, we should run a thread instead of using all CPUs. # |DocumentClassifier
		for my $model_spec (@model_spec) {
			my ($trfn, $tsfn, $modelfn) = split /\t/, $ttset ;
			do_train_test_wrapper($trfn, $tsfn, $modelfn, $model_spec, $result_file, (my $results_prefix = $model_spec) );
			++$do_train_test_serno;
		}
    } else {
# 		print STDERR "  >>> HERE2 \n";
		my $prun= ParallelRunner->new; # (1);
		for my $model_spec (@model_spec) {
			my ($trfn, $tsfn, $modelfn) = split /\t/, $ttset ;
			$prun->run(\&do_train_test_wrapper, $trfn, $tsfn, $modelfn, $model_spec, $result_file, (my $results_prefix = $model_spec) );
			++$do_train_test_serno;
		}
		$prun->wait();
	}
	
	my @retval = file($result_file) ;
	unlink $result_file;
	return @retval;
}




######################################################################################
sub do_cvtest_run_summarise(\@) {
    my @output = @{ $_[0] };
    
	my @summary;
	
	my %results ;

	my %convmat_tutti; 
	my @convmat_header; 
	
	for (grep { /\bConMat\b/ } @output) {
		my $line = $_; chomp $line ;
		my @parts = split /\t/, $line;
		my $dset = shift @parts;
		my $ConMat_dummy = shift @parts;
		my $label = shift @parts;
		my @values = @parts;
		if ( ! length($label) ) {
            @convmat_header = @values ;
		} else {
            $convmat_tutti{$label}{ $convmat_header[$_] } += $values[$_] for ( 0 .. $#convmat_header );
		}
	}
	
    push @summary, join("\t", "CV", "", @convmat_header)."\n";
	for my $i (@convmat_header) {
        push @summary, join("\t", "CV", $i, map { $convmat_tutti{$i}{$_} } @convmat_header)."\n";
	}
	
	for (grep { /Accuracy|AUC/ } @output) {
		my $line = $_; chomp $line ;
		my @parts = split /\t/, $line;
		my $dset = shift @parts;
		my $value = pop @parts;
		my $name = join "\t", @parts;
		push @{ $results{$name} }, $value ;
	}

	for my $label (sort keys %results) {
		my @res = @{ $results{$label} };
		if ( scalar(@res) >= 2 ) {
			my $mu = arith_mean(@res);
			my $sigma = se(@res); # / sqrt( scalar(@res) );
			push @summary, join("\t", "Mean", $label, $mu)."\n";
			push @summary, join("\t", "SE", $label, $sigma)."\n" if defined $sigma;
		} elsif ( scalar(@res) == 1 ) {
			push @summary, join("\t", $label, $results{$label}[0])."\n";
		} else {
			push @summary, join("\t", $label, 'NA')."\n";
		}
	}
    return @summary;
}

######################################################################################
sub write_to_file($\@) {
	my $filename = shift;
	my @filedata = @{ $_[0] };
	my $fh ;
	open ($fh, '>', $filename) or die "Could not write to '$filename' - $!";
	print $fh @filedata ;
	close $fh ;
}

######################################################################################
sub demultiplex_pipeline_args {
	my @args = @_;
	my @model_specs ;
	my @feature_proc_pipelines ;
	
	for my $arg (@args) {
		if ( $arg =~ /^PREP:\s*(.+)$/ ) {
			my $feature_proc_pipeline = $1;
			push @feature_proc_pipelines, $feature_proc_pipeline ;
		} else {
			push @model_specs, $arg;
		}
	}
	return (\@model_specs, \@feature_proc_pipelines);
}


######################################################################################
sub do_train_test_eval_all_classifiers(\@\@;$) { 
	my @tt_sets        = @{ $_[0] };
	my @model_specs    = @{ $_[1] };
	my $summary_prefix = $_[2];
	my @summary;

	if ( scalar(@tt_sets) == 1 and scalar(@model_specs) > 1 )  {
# 		print STDERR " >>> HERE \n";
		my @output = do_tttest_run_train_test(@model_specs, $tt_sets[0]);
# 		print STDERR map {" >>> $_ " } @output ;
		my %output_group ;  
		for (@output) {
			my ($model_spec, $results) = split /\t/, $_, 2;
			push @{ $output_group{$model_spec} }, $results;
		}
		for my $model_spec (@model_specs) {
			my @summary1 = map { (defined $summary_prefix ? "$summary_prefix | ": '')."$model_spec\t$_" } do_cvtest_run_summarise(@{ $output_group{$model_spec} });
			mlev_debug( MLEV::DL_MESSAGE, @summary1 );
			push @summary, @summary1 ;
		}
	} else {
		for my $model_spec (@model_specs) {
			my @output = do_cvtest_run_train_test($model_spec, @tt_sets);
			my @summary1 = map { (defined $summary_prefix ? "$summary_prefix | ": '')."$model_spec\t$_" } do_cvtest_run_summarise(@output);	
			mlev_debug( MLEV::DL_MESSAGE, @summary1 );
			push @summary, @summary1;
		}
	}
	return @summary;
}

######################################################################################
sub do_train_set_eval_all_classifiers_with_feature_prep_pl(\@\@\@) { 
	my @tt_sets        = @{ $_[0] };
	my @model_specs    = @{ $_[1] };
	my @fprep_specs    = @{ $_[2] };

	my @summary;
	my $fpcnt = 0;
	for my $feature_proc_pipeline (@fprep_specs) {
		++$fpcnt ;
		my $rubbish_bin_1 = FileGarbageBin->new;
		my @tt_sets_transformed;
		for my $i ( 0 .. $#tt_sets ) {
			my ($trfn, $tsfn) = split /\t/, $tt_sets[$i];

			my $fn_model   = "$tmpdir/mlev-$$-fp$fpcnt-ttset".($i+1)."-feature-proc.model";
			my $fn_tr_repl = "$tmpdir/mlev-$$-fp$fpcnt-ttset".($i+1)."-tr.txt";
			my $fn_ts_repl = "$tmpdir/mlev-$$-fp$fpcnt-ttset".($i+1)."-ts.txt";
		
			$rubbish_bin_1->add($fn_model) ;
			
			my @trfn_output = do_train($fn_model, $trfn, $feature_proc_pipeline);
			write_to_file($fn_tr_repl, @trfn_output);
			$rubbish_bin_1->add($fn_tr_repl) ;
			
			my @tsfn_output = do_predict($fn_model, $tsfn);
			write_to_file($fn_ts_repl, @tsfn_output);
			$rubbish_bin_1->add($fn_ts_repl) ;
			
			push @tt_sets_transformed, join("\t", $fn_tr_repl, $fn_ts_repl, $fn_model);
		}
		push @summary, do_train_test_eval_all_classifiers(@tt_sets_transformed, @model_specs, $feature_proc_pipeline);
		$rubbish_bin_1->empty();
	}
	return @summary;
}


######################################################################################
sub do_resampling_test { # Main subroutine: training-testing evaluation
	my @args = @_;
	my $type = shift @args;
	my $fn_data = shift @args; 

	my @tt_sets ;
	
	for ($type) {
		/rs/ and do { @tt_sets = file($fn_data); last; } ;
		my $split_spec = shift @args;
		/cv/ and do { @tt_sets = do_cvsplit($fn_data, $split_spec); last; } ;
		/tt/ and do { @tt_sets = do_split($fn_data,   $split_spec); last; } ;
	}

	my ($model_spec_ref, $feature_spec_ref) = demultiplex_pipeline_args(@args);
	my @model_specs = @{ $model_spec_ref };
	my @feature_proc_pipelines = @{ $feature_spec_ref };
		
# 	print @tt_sets;
	chomp for @tt_sets;
	
	my $rubbish_bin = FileGarbageBin->new;
	
	if ( $type ne 'rs' ) {
		$rubbish_bin->add( split(/\t/, $_) ) for @tt_sets ;
	}
    
	my @summary;
	
	if ( scalar(@feature_proc_pipelines) ) {
		@summary = do_train_set_eval_all_classifiers_with_feature_prep_pl(@tt_sets, @model_specs, @feature_proc_pipelines);
	} else {
		@summary = do_train_test_eval_all_classifiers(@tt_sets, @model_specs);
	}
	
	$rubbish_bin->empty();
	return @summary;
}


######################################################################################
sub do_tttest { # Main subroutine: training-testing evaluation
	my @args = @_;
	return do_resampling_test('tt', @args);
}

######################################################################################
sub do_cvtest { # Main subroutine: cross validation evaluation
	my @args = @_;
	return do_resampling_test('cv', @args);
}

######################################################################################
sub do_rstest { # Main subroutine: general resampling evaluation
	my @args = @_;
	return do_resampling_test('rs', @args);
}

######################################################################################
sub do_rstest_clean { # Main subroutine: general resampling evaluation
	my @args = @_;
	
	my $fn_data = shift @args; 

	my @tt_sets = file($fn_data);
	
	chomp for @tt_sets;
	
	my $rubbish_bin = FileGarbageBin->new;
	
	$rubbish_bin->add( split(/\t/, $_) ) for @tt_sets ;
	
	$rubbish_bin -> empty();
	
	return '';
}

######################################################################################
sub do_toarff { 
	my @args = @_;
	my $fn_data = shift @args; $fn_data = '-' if ! defined $fn_data ;
	my $tsv = load_primary_tsv($fn_data);
	print $tsv->to_arff();
	return '';
}
