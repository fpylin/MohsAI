#!/usr/bin/perl


package Classifier;

use strict;
use warnings;

use File::Basename;
use lib dirname (__FILE__);
use lib '.';

use Common;
use TSV;
use MLEV;
use SMLModel;
use Cwd;

my $__FILE_CWD__ = dirname(__FILE__);

use MIME::Base64;

my $cwd = getcwd;

our @EXPORT;
our @EXPORT_OK;

# Argument: $0 train model_file /training-set/ params ... 
# Argument: $0 test  model_file /test-set/
# Model should return "Class1\tClass2\nProbability1\tProbability2\n"
# training-set, usually in TSV format

##################################################################################
BEGIN {
    use Exporter   ();
    our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);
    $VERSION     = sprintf "%d.%03d", q$Revision: 1.1 $ =~ /(\d+)/g;
    @ISA         = qw(Exporter);
    @EXPORT      = qw(
		$action      $model_fn     $data_fn          @params
		$TSV_train   $TSV_test     $TSV_predictions
		$model            
		&register    &set_flag
		&set_class_label_levels
		@fields      $class_label  @class_levels
		&clean_training_data_in_situ
		&clean_test_data_in_situ
		&model_to_keep_after_training 
		&get_arff_file_for_training
		&get_arff_file_for_testing
		&weka_train  &weka_predict
		&match_feature_levels 
	);
    %EXPORT_TAGS = ();
    @EXPORT_OK   = qw();
}

sub v_safe { defined $_[0] ? $_[0] : 0; }

##################################################################################################
our $action = shift @ARGV;
our $model_fn = shift @ARGV;
our $data_fn = shift @ARGV;
our @params = @ARGV;
our $model = SMLModel->new($model_fn);

our $TSV_train = undef;
our $TSV_test = undef;
our $TSV_predictions = undef;

our $train_fun_ref = undef;
our $predict_fun_ref = undef;

our @fields ;
our $class_label = undef ;
our @class_levels;

our $weka_path = mlev_config( 'WEKA_PATH' ) // '/usr/local/weka' ; 
our $java_path = mlev_config( 'JAVA_PATH' ) // '/usr/bin/java'; 
our $weka_c = "$java_path -Xmx8192M -cp $weka_path";

our %flags;

##################################################################################################
sub register (&&) {
	$train_fun_ref = $_[0];
	$predict_fun_ref = $_[1];
}

##################################################################################################
sub set_flag($$) {
    my $varname = shift;
    my $value = shift;
    $flags{$varname}=$value;
}


##################################################################################################
sub get_class_label {
	my $cl = '';
	if ( defined $class_label ) {
		return $class_label;
	} elsif ( defined($model) and $model->has_attr('class')  ) { # and $model->has_attr('class')
		($cl) = $model->get_attr('class') ;
# 		print STDERR "!!!!$cl1\n\n\n";
		$cl =~ s/\t.*// if defined $cl;
	}
	
	if ( ! defined $cl or ! length($cl) ) { 
# 		print STDERR "####\n";
		$cl = $TSV_train->guess_class_label();
	} 
	return $cl;
}

##################################################################################################
sub set_class_label_levels {
	($class_label, @class_levels) = @_;
}


sub convert_class_labels_to_index {
	my $TSV = $_[0];
	my %class_level_index = map { $class_levels[$_] => $_ } (0..$#class_levels);
	for my $row ( @{ $TSV->{'data'} } ) {
		$$row{$class_label} = $class_level_index{ $$row{$class_label} }; 
	}
}

sub is_NA {
	my $x = shift;
	return ( (! defined $x ) || ( $x =~ /^(?:NA|\?|\s*)$/ ) );
}

##################################################################################################
sub clean_training_data_in_situ {
	my $params = shift;
	
    my $class_label = get_class_label();
#     my $ftmppath_model  = mlev_tmpfile("R-model.txt");
	my $ftmppath_src_tr = mlev_tmpfile("tr-dummified.tsv");

	my %missing_value_defaults;
	
    for my $field ( $TSV_train->fields() ) {
		
		my $missing_value_default = undef;
		
		if ( ! $TSV_train->is_numeric($field) ) { # categorical value
			my @non_NA_levels = grep { ! is_NA($_) } $TSV_train->levels($field);

			$TSV_train->remove( $field ) if scalar @non_NA_levels <= 1;
			
			# replace missing value by mode 
			my %histogram = $TSV_train->histogram($field);
			my $mode = argmax(%histogram);
			
			$missing_value_default = $mode;
#			print STDERR "$field\t[$mode]\tC\t".join(", ", scalar @non_NA_levels, @non_NA_levels)."\n";
			
		} else { # is numeric
			my @non_NA_values = grep { ! is_NA($_) } $TSV_train->get_column($field);
			my @uniq_non_NA_values = uniq( @non_NA_values );
			
			$TSV_train->remove( $field ) if scalar @uniq_non_NA_values <= 1;
			
			my $median = median(@non_NA_values);
			
			$missing_value_default = $median;
#			print STDERR "$field\t[$mean]\tN\t".join(", ", scalar @uniq_non_NA_values, @uniq_non_NA_values)."\n";
		}
		
		$missing_value_defaults{$field} = $missing_value_default ;
	}
	
	
    $TSV_train->remove_rows_by_criteria( sub { 
		my $h = $_[0];
# 		print map {"$class_label\t$_\t$$h{$_}\n" } keys %$h;
		return ( 
			(! exists $$h{$class_label}) || is_NA( $$h{$class_label} )
		) } 
	);
	
    $TSV_train->dummify( $class_label ); # dummify everything execept the class label
    
    $model->set_attr('features_dummified', @{ $TSV_train->{'fields'} });

#     print STDERR ( map { "||\t$_\t$missing_value_defaults{$_}\n" } keys %missing_value_defaults );
    $model->set_attr('feature_default_values', ( map { "$_\t$missing_value_defaults{$_}" } keys %missing_value_defaults) );
    
    convert_class_labels_to_index($TSV_train) if ( defined $params and $params->{'convert-class-labels-to-index'} );
    
	$TSV_train->save_as( $ftmppath_src_tr );
# 	print STDERR column( $TSV_train->to_string() );
    
    return $ftmppath_src_tr; # returns a temporary source file    
}



##################################################################################################
sub match_feature_levels {
	my @wanted_feature_levels = $model->get_attr( 'feature_levels' ); 

	my @wanted_features; 
	for (@wanted_feature_levels) {
        chomp;
        my ($f, @wanted_levels) = split /\t/, $_;
#       @data_feature_levels = $TSV_test->{'feature_levels'}{$f} ;
        push @wanted_features, $f;
        $TSV_test->{'feature_levels'}{$f}  = join("\t", @wanted_levels);
#       if join("\t",@data_feature_levels) ne join("\t",@wanted_levels)
	}
	
	my @features_test_set = (@wanted_features) ; # 
	print STDERR "Test set features: ".
		join("/", scalar(@{ $TSV_test->{'fields'} }), scalar(@features_test_set))." ".
		join(", ", ( map { /^b64\.(.*)/ ? "\"".decode_base64($1)."\" (\e[1;30m$_\e[0m)" : $_ } @features_test_set ) ).
		"\n";
	$TSV_test->{'fields'} = \@features_test_set ;
 }


##################################################################################################
sub clean_test_data_in_situ {
	my $params = shift;
    my $ftmppath_src_ts = mlev_tmpfile("ts-dummified.tsv");
    
#     $TSV_test->remove( $class_label ); 
# 	$TSV_test->{'features'} = $model->get_attr('feature_levels')
	
    $TSV_test->dummify( $class_label ) ;
    
    my @training_set_fields = $model->get_attr('features_dummified');
    
    $TSV_test->{'fields'} = \@training_set_fields ;

    for my $row (@{ $TSV_test->{'data'} }) {
		for my $f (@training_set_fields) {
			$$row{$f} = 0 if ! exists $$row{$f} ;
		}
    }
    
    convert_class_labels_to_index($TSV_test) if ( defined $params and $params->{'convert-class-labels-to-index'} );
#     print STDERR column( $TSV_test->to_string() );
    $TSV_test->save_as( $ftmppath_src_ts );
    
    return $ftmppath_src_ts; # returns a temporary source file    
}

##################################################################################################
our $ftmppath_trained_model = undef;

sub model_to_keep_after_training { # returns a temporary filename for importation
	my $extension = ( defined $_[0] ? $_[0] : 'model' );
    $ftmppath_trained_model = mlev_tmpfile( "ML.$extension" );
}


##################################################################################################
sub get_arff_file_for_training {
    my $arff_fn = mlev_tmpfile( "WEKA-train.arff" );
    my @arff_lines = $TSV_train->to_arff();
# 	print STDERR @arff_lines ;
    my @arff_attributes = grep { /\@attribute/ } @arff_lines ;
	$model->set_attr( 'weka_attributes', @arff_attributes ); 
    
    open  ARFF, ">$arff_fn";
    print ARFF @arff_lines ;
    close ARFF;
    return $arff_fn ;
}

##################################################################################################
sub get_arff_file_for_testing {
    my $arff_fn = mlev_tmpfile( "WEKA-test.arff" );
    my @arff_lines = $TSV_test->to_arff();
	print STDERR @arff_lines ;
    open  ARFF, ">$arff_fn";
    print ARFF @arff_lines  ;
    close ARFF;
    return $arff_fn ;
}

#########################################################################################
sub weka_train {
    my $classifier = shift;
    my $fn_model_output = shift;
    my $fn_arff_train = shift;
    my @classifier_params = @_;
    my $classifier_params = join(' ', @classifier_params);
    my $cmd = "$weka_c $classifier  -t $fn_arff_train -i -d $fn_model_output".(length($classifier_params)>0 ? " ".$classifier_params : '');
    system $cmd;
#     system "cat $fn_arff_train";
}

#########################################################################################
sub weka_predict {
    my $weka_classifier = shift;
    my $fn_trained_model = shift;
    my $fn_arff_test = shift;
    
    my @arff_attributes = grep { /\@attribute/ } file($fn_arff_test);
    my $attribute_class_str = pop @arff_attributes ;
    $attribute_class_str =~ s/\@attribute\s+//;
    my ($class_name, $levels_str) = ( $attribute_class_str =~ /(.*?)\s*{(.+)}/ );
    $class_name= trim($class_name);
    
    die "Class name mismatch! ''$class_label'' != ''$class_name'' !\n" if $class_label ne $class_name ;
    
    my @arff_class_levels = map { s/^"(.+)"$/$1/; $_ } split /\s*,\s*/, $levels_str;
    
    my $weka_cmd = "$weka_c $weka_classifier -p 0 -distribution -l $fn_trained_model -T $fn_arff_test";
    my @weka_output = qx {$weka_cmd};

    my ($header_line) = grep { /inst.*actual.*predicted.*error.*distribution/ } @weka_output ;
    
    @weka_output = grep { /\d/ and ! /==.*==|^\s*$/} @weka_output ;
    
    my ($lentoheader) = ( $header_line =~ /^(.*)distribution/ );  $lentoheader = length($lentoheader);

    for my $line (@weka_output) {
        chomp $line;
        my $distribition = substr $line, $lentoheader;
        my @distribution = split /\s*,\s*/, $distribition ;
        s/\*// for @distribution ;
        my %a = map { $arff_class_levels[$_] => $distribution[$_] } (0..$#distribution);
        $TSV_predictions->push_rows(\%a);
    }
}

##################################################################################################
sub rectifying_test_set_features {
	my @model_fields ;
	
    my %missing_value_defaults ;
    
    if ( $model->has_attr('feature_default_values') ) {
		%missing_value_defaults = map { chomp; my ($f, $v) = split /\t/, $_; $f => $v } $model->get_attr('feature_default_values');
	}
	
    for my $l ( $model->get_attr( 'feature_levels' ) ) {		
# 		print STDERR ">>$$>> $l\n";
        my ($f, @z) = split /\t/, $l;
        push @model_fields, $f;
        $TSV_test->{'feature_levels'}{$f} = join("\t", @z);
#         next if scalar(@z) and $z[0] eq 'NUMERIC';
# 			print STDERR ">> $f\n";
        my %z = map { $_ => 1 } @z;
        my $rowno = 0;
        for my $row (@{ $TSV_test->{'data'} } ) {
			my @imputed;
			++$rowno;
			if ( ! exists ${$row}{$f} and ($f ne $class_label) )  {
				if ( exists $missing_value_defaults{$f} ) {
					${$row}{$f} = $missing_value_defaults{$f};
					push @imputed, "$f=$missing_value_defaults{$f}";
				} else {
					warn "Test set fields: ".join(" | ", sort keys %{ $row })."\n";
					die "Classifier.pm: rectifying_test_set_features(): Row $rowno: Feature ''$f'' not present in test set!" ;
				}
			}
			next if scalar(@z) and $z[0] eq 'NUMERIC';
			# printf STDERR ">$f: ${$row}{$f}\n" if ! exists $z{ ${$row}{$f} } ;
			my $v = ${$row}{$f} // '';
			warn "Classifier.pm: rectifying_test_set_features(): Row $rowno: imputed missing values: ".join(", ", @imputed).".\n" if scalar(@imputed);
            warn "Classifier.pm: rectifying_test_set_features(): Row $rowno: \e[1;31mCategorical feature not found in test set: $f=$v\e[0m\n" if ! exists $z{ $v } and ( $v ne 'NA' ) and ( $v ne '?' ) ;
            ${$row}{$f} = 'NA' if exists $$row{$f} and ! exists $z{ ${$row}{$f} } ;
        }
    }
#     $TSV_test->{'fields'} = \@model_fields;
#     print STDERR "Classifier::rectifying_test_set_features(): ".join("\t", "$class_label:", @class_levels)."\n";
    
	$TSV_test->{'feature_levels'}{$class_label} = join("\t", @class_levels);
}



##################################################################################################
sub run {
# 	print STDERR "\e[1;35mAction $action\e[0m\n";

	(warn "$0: No action specified.\n" and return 0) if ! defined $action;
	
	if ( $action eq 'train' ) {
		# register the model parameter
# 		print STDERR "\e[1;35mModel Class\e[0m: ".$model->get_attr('class')."\n";
		
		$model->set_spec( join(" ", $0, @params) );  
		
		(warn "$0: Training file not specified.\n" and return 0) if ! defined $data_fn;
		(warn "$0: Training file ''$data_fn'' not readable.\n" and return 0) if ! -f $data_fn;
	
		$TSV_train = new TSV($data_fn);
		@fields = $TSV_train->fields();
		$class_label = get_class_label();
		
		if ( ! exists $flags{'do_not_rectify_prediction_data'} ) { 
			$TSV_train->enumerate_all_feature_levels() ;
		} else {
			$TSV_train->enumerate_feature_levels($class_label) ;
		}
		
		@class_levels = grep { ! is_NA($_) } split(/\t/, $TSV_train->{'feature_levels'}{$class_label});
		
		$model->set_attr( 'class', join("\t", $class_label, @class_levels) ); 
		$model->set_attr( 'feature_levels', map { join("\t", $_, v_safe($TSV_train->{'feature_levels'}{$_}) ) } @fields ); 

		&$train_fun_ref();
		wait;
		    
		$model->set_attr( 'class', join("\t", $class_label, @class_levels) ); 
		$model->set_attr( 'feature_levels', map { join("\t", $_, v_safe($TSV_train->{'feature_levels'}{$_}) ) } @fields ); 
		
        # import the actual model file 
#         system "ls -l $ftmppath_trained_model";
        $model->import_model_from_file($ftmppath_trained_model) if defined $ftmppath_trained_model and -f $ftmppath_trained_model ;
        
        # save the TAR file 
		$model->save();
		return '';
	}
    
	if ( $action eq 'predict' ) {
		$TSV_test = new TSV($data_fn);
		$TSV_predictions = new TSV;
		
		my ($class_str) = $model->get_attr( 'class' );
		($class_label, @class_levels) = split /\t/, $class_str;
		$TSV_predictions->set_fields(@class_levels);
		
		rectifying_test_set_features() if ! exists $flags{'do_not_rectify_prediction_data'};

		match_feature_levels();
		
		my $fn_output = &$predict_fun_ref() ;
		wait;
		
# 		print STDERR "Output file: $fn_output\n";
# 		print STDERR map { chomp; s/\s+$//g; s/\t/\|/g; "[$_]\n" } file($fn_output);
		
		if ( defined($fn_output) and ( -f $fn_output) ) {
            $TSV_predictions = new TSV;
            my @prediction_lines = file($fn_output);
            @prediction_lines = map { s/\s+$//; $_ } @prediction_lines;
# 			print STDERR map { chomp; s/\t/\|/g; "[$_]\n" } @prediction_lines ;
            $TSV_predictions->import_data(@prediction_lines);
		} 
		
# 		wait;
		print $TSV_predictions->to_string();
#         printf "%s", "A".$str."B";
#         STDERR "LALALA". $zzz1 . "ALALA";
        
		return 1;
	}

	warn "$0: Invalid action ''$action''\n";
	return '';
}

##################################################################################################
END {
	run();
	wait;
}

##################################################################################################
1;
