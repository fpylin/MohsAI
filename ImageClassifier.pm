#!/usr/bin/perl

package ImageClassifier;

use strict;
use warnings;

use File::Copy;
use File::Type;
use Cwd;
use POSIX;

use File::Basename;
use lib dirname (__FILE__);
my $__FILE_CWD__ = dirname(__FILE__);

use ParallelRunner;
use ImageUtils;
use Digest::SHA1;

use SMLModel;
use MLEV;

my $mlev_dir = dirname (__FILE__);
my $cwd = getcwd;

use Common;
use TSV;

use threads;

our @EXPORT;
our @EXPORT_OK;

##################################################################################
BEGIN {
    use Exporter   ();
    our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);
    $VERSION     = sprintf "%d.%03d", q$Revision: 1.1 $ =~ /(\d+)/g;
    @ISA         = qw(Exporter);
    @EXPORT      = qw();
    %EXPORT_TAGS = ();
    @EXPORT_OK   = qw();
}

my $fn_casefile = "/tmp/imgclassify-caseindex-$$.tsv";
my $fn_augdir = "/tmp/imgclassify-augdata";

mkdir $fn_augdir if ! -d $fn_augdir;

sub mkabspath { 
	my $x = shift; 
	return ( ( ( $x =~ /^\// ) ? $x : $cwd.'/'.$x ) =~ s/\/(?:\.?\/)+/\//gr );
}


our $file_type = File::Type->new;

my $image_classifier = 'CL-TFImage'; 

###############################################################################
sub new {
    my ($class, $fn_model, $classifier) = @_;
    my $self = { 'fn_model' => $fn_model };
    
    $self->{'classifier'} = defined $classifier ? $classifier : $image_classifier ;
    
    bless($self, $class);
    
    return $self;
}


###############################################################################
sub run {
	my $cmd = shift;
	print STDERR "\e[1;37m$cmd\e[0m\n";
 	system $cmd;
}

###############################################################################
sub make_case_file_data($$) {
	my $self = shift;
	my $fn_src = shift;
	my $score = shift;
	
	return join("\t", $fn_src, $score)."\n";
}

###############################################################################
sub gen_caseindex {
	my @casefile = @_;
	open CASEFILE, ">$fn_casefile" or die "Unable to write case file ''$fn_casefile''";
	print CASEFILE "image\tclass\n";
	print CASEFILE @casefile;
	close CASEFILE;
	return $fn_casefile;
}


###############################################################################
sub retrain {
	my $self = shift;
	my $params = shift;
	my $fn_model = $self->{'fn_model'};
	my $trained_model =  SMLModel->new($fn_model);
	my @posex = $trained_model->get_attr('posex');
	my @negex = $trained_model->get_attr('negex'); 
		
	my ($pipeline) = $trained_model->get_attr('pipeline');
	my $pipeline_o = $pipeline;
	for my $arg ( keys %{ $params } ) {
		next if $arg !~ /^(?:arch|epochs?|split-tiles|patience|trainable|arch|hidden-layers|batch-size)$/;
		$pipeline =~ s/\b$arg=\S+//g;  
		$pipeline .= " $arg=$params->{$arg}";
		$pipeline =~ s/  +/ /g;
	}
	
	$self->{'classifier'} = $params->{'pipeline'} = $params->{'classifier'} = $pipeline;
	print STDERR "Pipeline modified: $pipeline_o --> $pipeline\n" if $pipeline_o ne $pipeline;
	print STDERR "Retraining pipeline: $pipeline\n";

	$params->{'posex'} = \@posex;
	$params->{'negex'} = \@negex;
	return $self->train($params);
}

###############################################################################
sub add_posex {
	my $self = shift;
	my @casefiles = @_;
	my $fn_model = $self->{'fn_model'};
	my $trained_model =  SMLModel->new($fn_model);
	my @posex = $trained_model->get_attr('posex') if $trained_model->has_attr('negex') ;
	push @posex, (map { mkabspath($_) } @casefiles);
	@posex = sort uniq @posex ;
	$trained_model->set_attr('posex', @posex);
	$trained_model->save();
}

###############################################################################
sub add_negex {
	my $self = shift;
	my @casefiles = @_;
	my $fn_model = $self->{'fn_model'};
	my $trained_model =  SMLModel->new($fn_model);
	my @negex ;
	@negex = $trained_model->get_attr('negex') if $trained_model->has_attr('negex') ;
	push @negex, ( map { mkabspath($_) } @casefiles);
	$trained_model->set_attr('negex', @negex) ;
	$trained_model->save();
}

###############################################################################
sub process_cases ($\@;$) {
	my $params = $_[0];
	my @cases = @{ $_[1] };
	my $lookup = $_[2];
	
	@cases = uniq(@cases);
	my @cases1;
	for my $case (@cases) {
# 		print STDERR "ImageClassifier.pm: process_cases(): case = $case --\n";
		my ($file, $page) = ( $case =~ /^(.+?)(?::(\d+))?$/ );
# 		print STDERR "ImageClassifier.pm: process_cases(): $file : $page \n" if defined $page;
		for ( $file_type->mime_type($file) ) {
			if ( ! -f $file ) {
				print STDERR "ImageClassifier.pm: process_cases(): $file is not readable\n";
				next;
			}
			/image\/(?:jpeg|png)/ and do { push @cases1, $file; last; } ;
			/application\/pdf|x-pdf/ and do { 
				my $file_sha1 = ImageUtils::get_file_sha1_b16($file);
				my $pagelim = defined $page ? " -f $page -l $page " : '';
				my @dirent = dirr($fn_augdir) ;
				if ( ! grep { /$file_sha1/ } @dirent ) {
					my $cmd = "/usr/bin/pdftoppm -jpeg -r 96 $pagelim '$file' $fn_augdir/$file_sha1";
					print STDERR "\e[1;37m>$cmd\e[0m\n";
					system $cmd;
					@dirent = dirr($fn_augdir) ;
				}
				
				for my $e ( grep { /$file_sha1/ } @dirent ) {
					my ($pageno) = ( $e =~ /-0*(\d+)\.jpe?g$/i );
					$$lookup{$e} = $file."__pdfpage".$pageno; 
					push @cases1, $e; 
				}
				last;
			} ;
		}
	}
	
	@cases = sort @cases1 ;
	
	sub run_jpeginfo {
		my $thread_id = shift;
		my @filelist = @_;
		my $jpeglist_tmp_fn = mlev_tmpfile("jpeglist-$thread_id");
		open JPEGINFO, ">$jpeglist_tmp_fn";
		print JPEGINFO map {"$_\n"} @filelist;
		close JPEGINFO;
		my @errlist = map { chomp; my $f = $_; $f =~ s/^(.+?\.(?i:jpe?g))  .*/$1/; print STDERR "ERROR in $f. Removed.\n"; $f } grep { ! /OK/ } qx{/usr/bin/jpeginfo -c -f $jpeglist_tmp_fn};
		unlink $jpeglist_tmp_fn;
		return @errlist ;
	}
	
	if ( -x '/usr/bin/jpeginfo' ) {
		my $ncpus = `nproc`; chomp $ncpus ;
		my $batch = int ( scalar(@cases) / $ncpus ) + 1;
		my %errlist;
		if ( $batch < 32 )  {
			%errlist = map { $_ => 1 } run_jpeginfo(0, @cases);
		} else {
			my @to_check = @cases;
			my $cnt = 0;
			while (scalar(@to_check) > 0) {
				my @batch1 = splice @to_check, 0, min($batch, scalar(@to_check));
				my ($thr) = threads->create(\&run_jpeginfo, $cnt, @batch1);
				++$cnt;
			}
			for my $thr (threads->list()) {
				my @errlist = $thr->join();
				$errlist{$_} = 1 for (@errlist);
			}
		}
		@cases = grep { ! exists $errlist{$_} } @cases ;
	}
	
	if ( exists $params->{'split-tiles-dim'} and $params->{'split-tiles-dim'} =~ /^(\d+)x?(\d+)?$/ ) {
		($params->{'split-tiles-w'}, $params->{'split-tiles-h'}) = ($1, $2 // $1);
	}
	
	if ( exists $params->{'input_dim'} and $params->{'input_dim'} =~ /^(\d+)x?(\d+)?$/ ) {
		($params->{'input_w'}, $params->{'input_h'}) = ($1, $2 // $1);
	}
	
	my $iw = $params->{'split-tiles-w'} // 224;
	my $ih = $params->{'split-tiles-h'} // 224;
	my $zw = $params->{'input_w'} // 224;
	my $zh = $params->{'input_h'} // 224;
	my @scales = (1.0);
	
	if ( defined $params->{'augscales'} ) {
		@scales = split /,/, $params->{'augscales'};
		print STDERR "Images will be augmented by scaling: ".join(", ", (map { "$_ x" } @scales) )."\n";
	}
	
	# image augmentation options
	if ( defined $params->{'split-tiles'} ) {
		if ( defined $lookup ) {
			my @case1; 
			for my $case (@cases) {
				my @z = ( gen_tiles( {
							image_file => $case, 
							tile_width => $iw, 
							tile_height => $ih, 
							stride_factor => ($params->{'split-tiles-stride'}//1.0), 
							output_dir => $fn_augdir, 
							size_w => $zw, 
							size_h => $zh,
							scales => \@scales
							} ) 
						); # $case, 
				$$lookup{$_} = ( exists $$lookup{$case} ?  $$lookup{$case} : $case ) for @z;
				push @case1, @z;
			}
			@cases = @case1;
# 			print STDERR "\e[38;5;86m1: $_\e[0m\n" for @cases;
		} else {
			@cases = map { (gen_tiles( {
								image_file => $_, 
								tile_width => $iw, 
								tile_height => $ih, 
								stride_factor => ($params->{'split-tiles-stride'}//1.0), 
								output_dir => $fn_augdir, 
								size_w => $zw, 
								size_h => $zh,
								scales => \@scales
								} )
							) 
						} (@cases);  # $_, 
# 			print STDERR "\e[38;5;66m2: $_\e[0m\n" for @cases;
		}
	}

	if ( defined $params->{'split-tiles'} and defined $params->{'dewhite-model'} ) {
		my $dewhite_model_fn = $params->{'dewhite-model'};
		my $tmpfn = "/tmp/imgclassify-$$.tmp";
		open TMPFN, ">$tmpfn";
		print TMPFN map { "$_\n" } @cases;
		close TMPFN;
		my @output = qx{ $0 predict $dewhite_model_fn $tmpfn | tee $tmpfn.results };
		my @cases1 = ();
		for (@output) {
			chomp;
			my ($fn, $class_is_white, $white_prob_str) = split /\t/, $_, 3;
			my ($white_prob) = ( $white_prob_str =~ /Y:(\S+)/ );
			next if $white_prob > 0.90;
			push @cases1, $fn;
		}
		unlink $tmpfn;
		@cases = @cases1 ;
	}
	
	return @cases ;
}

sub mk_classifier_str($) {
	my $self = shift;
	my $params = shift;
	my $retval = $self->{'classifier'} ; 
# 	print "\e[1;31m". join(" ", keys %{ $params }). "\e[0m";
	for my $arg ( qw(epochs patience layers hidden-layers batch-size arch) ) {
		$retval .= ( ( defined $params->{$arg} ) ? " $arg=".$params->{$arg} : "" );
	}
	for my $arg ( qw(trainable) ) {
		$retval .= ( ( defined $params->{$arg} ) ? " $arg" : "" );
	}
	return $retval ;
}

###############################################################################
sub train {
	my $self = shift;
	my $params = shift;
	my $fn_model = $self->{'fn_model'};
	my @posex = @{ $params->{'posex'} };
	my @negex = @{ $params->{'negex'} };

	my %lookup_table;
	
	@posex = process_cases($params, @posex, \%lookup_table); 
	@negex = process_cases($params, @negex, \%lookup_table);
	
	my @posex_lines = ( map { $self->make_case_file_data($_, 'Y') } @posex);
	my @negex_lines = ( map { $self->make_case_file_data($_, 'N') } @negex);
	
	my $fn_casefile = gen_caseindex( @posex_lines, @negex_lines );
	
	unlink $fn_model if -f $fn_model;

	$self->{'classifier'} = $params->{'classifier'} if defined $params->{'classifier'} ;
	
	my $cmd = "$__FILE_CWD__/mlev.pl train $fn_model $fn_casefile ";
	
	$cmd .= "'".($self->mk_classifier_str($params))."'";
	
	run $cmd; 

	my $trained_model = SMLModel->new($fn_model);
	$trained_model->set_attr('posex', ( map { mkabspath( $lookup_table{$_} // $_ ) =~ s/__pdfpage/:/r } @posex ) );
	$trained_model->set_attr('negex', ( map { mkabspath( $lookup_table{$_} // $_ ) =~ s/__pdfpage/:/r } @negex ) );
	$trained_model->save();
	
	unlink $fn_casefile;
}

###############################################################################
sub cvtest {
	my $self = shift;
	my $params = shift;
	my @posex = @{ $params->{'posex'} };
	my @negex = @{ $params->{'negex'} } ;
	
	$self->{'classifier'} = $params->{'classifier'} if defined $params->{'classifier'} ;
	
	@posex = process_cases($params, @posex); 
	@negex = process_cases($params, @negex);
	
	my @posex_lines = ( map { $self->make_case_file_data($_, 'Y') } @posex);
	my @negex_lines = ( map { $self->make_case_file_data($_, 'N') } @negex);
	
	my $fn_casefile = gen_caseindex( @posex_lines, @negex_lines );
	
	my $folds = $params->{'cv-folds'};
	
	run "$__FILE_CWD__/mlev.pl cvtest $fn_casefile $folds '".mk_classifier_str($params)."'";
}


sub safe_NA { return ( $_[0] eq 'NA' ? 0 : $_[0] ) }

sub entropy {
	my $p = shift;
	return 0 if $p >= 1 or $p <= 0;
	my $q = 1 - $p;
	return ( - ( $p * log($p) ) - ( $q * log($q) ) ) / log(2) ;
}

sub infogain  {
	my $p = shift;
	my $p0 = shift;
	return entropy($p0) - entropy($p);
}



sub mean_infogain {
	my @v = @_;
	my $p0 = median(@v);
	my @ig = map { infogain($_, $p0) } @v;
# 	print STDERR "\e[1;35m".join("\t", @ig)."\e[0m\n";
	return mean(@ig);
}

###############################################################################
sub predict {
	my $self = shift;
	my $params = shift;
	my $fn_model = $self->{'fn_model'};

	my @test_cases = map { chomp; $_ } @{ $params->{'cases'} } ;
	my @test_cases0 = @test_cases ;
	
	my %fn_orig;
	
# 	print STDERR "\e[38;5;126mF: BEFORE\t\e[0m\n" for @test_cases;
	@test_cases = process_cases($params, @test_cases, \%fn_orig);
# 	print STDERR "\e[38;5;106mF: AFTER\t$_\e[0m\n" for @test_cases ;
# 	if (scalar(%fn_orig)) {
# 		print STDERR "\e[38;5;106mF: MAPPING\t$_ => $fn_orig{$_}\e[0m\n" for @test_cases ;
# 	}

	my $fn_casefile = gen_caseindex( ( map { $self->make_case_file_data($_, '?') } @test_cases ) );

	my @pred_output = qx{$__FILE_CWD__/mlev.pl predict $fn_model $fn_casefile};

	my $predictions = TSV->new;
	
	if (! scalar(@pred_output) ) {
		my @casefiles = file($fn_casefile);
		print STDERR map { "\e[38;5;92m$casefiles[$_]\e[0m" } (0..$#casefiles) ;
	}
	$predictions->import_data(@pred_output);
# 	print STDERR $predictions->to_string();
	
	my @output ;

	my %orig_tc = (); # original test class, testing score
	my %orig_pc = (); # original predicted class
	my %transformed_images; 
	
	for my $i (0..$#test_cases) {
	    my $tc = $test_cases[$i];
		my %row = defined $predictions->{'data'}[$i] ? %{ $predictions->{'data'}[$i] } : ();
		my $class = argmax(%row);
		my $prob_str = join("\t", map { "$_:$row{$_}" } sort { safe_NA( $row{$b} ) <=> safe_NA( $row{$a} ) } keys %row);
		push @output, join("\t", ( exists $fn_orig{$tc} ? "$fn_orig{$tc}:$tc" : $tc ), $class // '?', $prob_str)."\n";
		if ( exists $fn_orig{$tc} ) {
			my $fn_orig_tc = $fn_orig{$tc} ;
			$transformed_images{ $fn_orig_tc }{ $tc } = 1;
			$orig_pc{ $fn_orig_tc }{ $class } ++ if defined $class ;
			for my $cl ( keys %row ) {
				push @{ $orig_tc{ $fn_orig_tc }{$cl} }, $row{$cl};
			}
# 			print STDERR "Transformed: [$fn_orig{$tc}] --> [$tc]\n";
#			print STDERR "orig_pc: $fn_orig_tc  [$tc] --> [$tc]\n";
		}
	}

	sub predict_split_tiles_metric($\&\&\@\%) {
	    my $metric      = $_[0];
	    my $metric_func = $_[1];
	    my $aggreg_func = $_[2];
	    my @tcases      = @{ $_[3] }; # test cases;
	    my %orig_tc     = %{ $_[4] } ;
	    my @retval ;
	    
#	print STDERR "$metric: ".join(" ", @tcases)."\n";
	    for my $i (0 .. $#tcases) {
		my $tc = $tcases[$i];
		next if ! exists $orig_tc{$tc};
		my $nmax = 0;
		my %metric = map { my $c = $_; my @v = @{ $orig_tc{$tc}{$c} }; $nmax = max($nmax, scalar(@v)); $c => $metric_func->(@v) } keys %{ $orig_tc{$tc} };
		next if $nmax <= 1;
#		print STDERR "$i $tc ".scalar(keys %metric)." ".join(" ", sort keys %metric)."\n";
		my $class = $aggreg_func->(\%metric);
		my $prob_str = join("\t", map { "$_:$metric{$_}" } sort { $metric{$b} <=> $metric{$a} } keys %metric);
#		print STDERR "$metric: [$tc]\t".join("\t", "$tc:$metric", $class, $prob_str)."\n";
		push @retval, join("\t", "$tc:$metric", $class, $prob_str)."\n";
	    }
	    return @retval;
	}
	
	if ( scalar(%orig_tc) ) {
	    my %orig_tc_r ;
	    for my $f (uniq( values %fn_orig) ) {
		my $k = ($f =~ s/__pdfpage\d+//r);
#		print STDERR "$k \t $f \n";
		next if $k eq $f;
		$orig_tc_r{ $k }{$f} = 1;
	    }
#	    print STDERR "B $_\n" for @test_cases0;
	    @test_cases0 = map { exists $orig_tc_r{$_} ? (sort keys %{ $orig_tc_r{$_} }) : $_  } @test_cases0;
#	    print STDERR "A $_\n" for @test_cases0;
		
	    push @output, predict_split_tiles_metric( 'median', &median, &argmax, @test_cases0, %orig_tc );
	    push @output, predict_split_tiles_metric( 'max',    &max,    &argmax, @test_cases0, %orig_tc );
	    push @output, predict_split_tiles_metric( 'min',    &min,    &argmax, @test_cases0, %orig_tc );
# 		my $N = scalar(@test_cases0);
		for my $i (0..$#test_cases0) {
			my $tc = $test_cases0[$i];
			next if ! exists $orig_pc{$tc};
			my $N = scalar( keys %{ $transformed_images{$tc} } );
			my %vote = map { $_ => ( ($orig_pc{$tc}{$_}) / $N) } ( keys %{ $orig_pc{$tc} } );
			next if scalar (keys %vote) == 1;
# 			for my $c ( keys %{ $orig_pc{$tc} } ) {
# 				print "$c\t".join("\t", $orig_pc{$tc}{$c})."\t".$N."\n";
# 			}
			my $class = argmax(%vote);
			my $prob_str = join("\t", map { "$_:$vote{$_}" } sort { $vote{$b} <=> $vote{$a} } keys %vote);
			push @output, join("\t", "$tc:vote", $class, $prob_str)."\n";
		}
		for my $i (0..$#test_cases0) {
			my $tc = $test_cases0[$i];
			next if ! exists $orig_tc{$tc};
			next if scalar @{ $orig_tc{$tc}{'Y'} } == 1;
			my $mean_infogain = mean_infogain( @{ $orig_tc{$tc}{'Y'} } ) ;
			my $class = '?';
			push @output, join("\t", "$tc:infogain", $class, $mean_infogain)."\n";
		}
		
	}
	
	return @output ;
}



#####################################################################################
END {
    run "rm -rf $fn_casefile" if -f $fn_casefile;
};

1;
