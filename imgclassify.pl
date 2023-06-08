#!/usr/bin/perl

use strict;
use warnings;

use Getopt::Long;
use Cwd;
use File::Basename;
use lib dirname (__FILE__);
use Common;
use ImageClassifier;
my $__FILE_CWD__ = dirname(__FILE__);

# my $fn_model = undef;
my $classifier_name ='CL-TFImage';
my $imgc = undef;
my $cwd = getcwd;

sub mkstem { 
	my $x = shift;
	$x =~ s|.*/||;
	$x =~ s|\.model$||;
	return $x;
}

my %params = ();


if (scalar(@ARGV)) {

	GetOptions ( \%params, "split-tiles", "no-split-tiles", "split-tiles-stride=s", "split-tiles-dim=s", "dewhite-model=s", "augscales=s", "epochs=i", "patience=i", "arch=s", "hidden-layers=s", "batch-size=i", "trainable", "not-trainable");
	
	my @args = @ARGV;
            
	my $command = shift @args;
	my $fn_model = shift @args;
	
	die "$0: FATAL: no command specified." if ! $command ;
	if ( $command !~ /\+|\*|cvtest/ ) {
		die "$0: FATAL: model file is not specified.\n" if ! defined $fn_model;
		die "$0: FATAL: ''$fn_model'' is not a model file.\n" if $fn_model !~ /\.model$/;
		die "$0: FATAL: model file ''$fn_model'' does not exist.\n" if $command =~ /predict/ and ! -f $fn_model;
		$imgc = ImageClassifier->new($fn_model);
	}
	
	if ($command =~ /cvtest/) {
		$imgc = ImageClassifier->new;
		unshift @args, $fn_model;
	}
	
	for ($command) {
		/^train$/             and do { exit do_TRAIN(@args);                                      };
		/^retrain$/           and do { exit do_RETRAIN($fn_model, @args);                         };
		/^predict$/           and do { print do_PREDICT(@args);                           exit 0; };
		/^cvtest/             and do { exit do_CVTEST(@args);                                     };
		/^add-posex$/         and do { exit do_ADD_POSEX($fn_model, @args);                       };
		/^add-negex$/         and do { exit do_ADD_NEGEX($fn_model, @args);                       };
		/^add-examples$/      and do { exit do_ADD_EXAMPLES($fn_model, @args);                    };
	}
}

die "Usage: $0 [train|predict|predict_text]+? classifier.model files ...\n";

sub rectify_path { my @a = @_ ;
# 	print STDERR map { "\e[38;5;76mrectify_path: $_\e[0m\n" } @a;
	@a = map { /^\// ? $_ : $cwd.'/'.$_ } 
		grep { /(^.*\.(?:jpe?g|bmp|png|pdf))(?::\d+)?$/i and ( -f $1 ) } 
		map { 
			-d $_ ? dirr($_) : 
			/^list:(.+)/ ? ( map {chomp; $_ } file($1) ) : # file contains a list
			$_ 
			} 
		@a;
	return @a;
	}

sub do_PREDICT {
	my @cases = @_;
	
# 	print STDERR "\e[38;5;200mdo_PREDICT($cases[0], ...)\e[0m\n";
	
# 	print STDERR "do_PREDICT(...): ".scalar(@cases)." cases.\n".join("\n",@cases)."\n";
	
	if ( scalar(@cases) == 1 and -f $cases[0] and $cases[0] !~ /jpe?g$|png$|pdf(?::\d+)?/i ) { # this means annoation file is supplied
# 		print STDERR "do_PREDICT($cases[0], ...): annoation file supplied.\n";
		my $casefile = shift @cases;
# 		my ($posex, $negex) = file($casefile);
		for ( file($casefile) ) {
			chomp;
			my @parts = split /\t/, $_;
			my $filename = pop @parts;
			push @cases, $filename;
		}
	}
	
# 	print STDERR "\e[38;5;150mdo_PREDICT(...): Cases: \e[0m\n" ;
	@cases = rectify_path @cases ;
# 	print STDERR "\e[38;5;150mdo_PREDICT(...): $_\e[0m\n" for @cases;
# 	print map { "$_\n" } @cases;
	return $imgc->predict( { 'cases' => \@cases, %params } );
}

sub process_casefile {
	my $casefile = shift;
	my @posex;
	my @negex;
	for ( file($casefile) ) {
		chomp;
		my ($score, $path) = split /\t/, $_;
		my @cases = rectify_path($path);
		print STDERR "$score|$path|$_\n" for @cases;
		if ( $score =~ /^[0N]$/ ) {
			push @negex, @cases;
		} else {
			push @posex, @cases;
		}
	}
# 	print "$casefile: posex:".scalar(@posex)." negex:".@negex."\n";
	return (\@posex, \@negex);
}

sub do_ADD_EXAMPLES {
	my $fn_model = shift ;
	my $casefile = shift ;
	my ($posex, $negex) = process_casefile($casefile);
	my @posex = @{ $posex };
	my @negex = @{ $negex };
	my $imgc = ImageClassifier->new( $fn_model );
	print STDERR "Adding ".scalar(@posex)." positive examples.\n";
	$imgc->add_posex(@posex);
	print STDERR "Adding ".scalar(@negex)." negative examples.\n";
	$imgc->add_negex(@negex);
}

sub do_ADD_POSEX {
	my $fn_model = shift;
	my $imgc = ImageClassifier->new( $fn_model );
	my @cases = rectify_path(@_);
	print STDERR "Adding ".scalar(@cases)." positive examples.\n";
	$imgc->add_posex(@cases);
}

sub do_ADD_NEGEX {
	my $fn_model = shift;
	my $imgc = ImageClassifier->new( $fn_model );
	my @cases = rectify_path(@_);
	print STDERR "Adding ".scalar(@cases)." negative examples.\n";
	$imgc->add_negex(@cases);
}

sub do_RETRAIN {
	my $fn_model = shift;
	my @args = @_;
	if (scalar(@args) == 1) { # this means annoation file is supplied
		my $casefile = shift @args;
		do_ADD_EXAMPLES($fn_model, $casefile);
	}
	
	my $imgc = ImageClassifier->new( $fn_model );
	$imgc->retrain( \%params );
}

sub load_cases_from_args {
	my @args = @_;
	my %retval ;
	print STDERR "process_args(): ".scalar(@args)." arguments: ".join(' ', @args)."\n";
	if (scalar(@args) == 1) { # this means annoation file is supplied
		my $casefile = shift @args;
		my ($posex, $negex) = process_casefile($casefile);
		$retval{posex} = $posex ;
		$retval{negex} = $negex ;
	} else {
		print STDERR join(' | ',@args)."\n";
		while ( scalar(@args) ) {
			my $a = shift @args;
			last if $a eq '--';
			last if $a eq 'NEGEX:';
			push @{ $retval{posex} }, $a;
		}
		push @{ $retval{negex} }, @args;
	}
	
	die "No positive examples specified" if ! defined $retval{posex} or ! scalar @{ $retval{posex} };
	die "No negative examples specified" if ! defined $retval{negex} or ! scalar @{ $retval{negex} };
	
	@{ $retval{posex} } = rectify_path(@{ $retval{posex} });
	@{ $retval{negex} } = rectify_path(@{ $retval{negex} });

	print STDERR scalar(@{ $retval{posex} })." positive examples specified.\n" ;
	print STDERR scalar(@{ $retval{negex} })." negative examples specified.\n" ;

	my %posex = map { $_ => 1 } @{ $retval{posex} } ;
	@{ $retval{negex} } = grep { ! exists $posex{$_} } @{ $retval{negex} } ;

	return %retval ;
}

sub do_TRAIN {
	my @args = @_;
	
	my %caselist = load_cases_from_args(@args);
	
	$imgc->train( { 
		'posex' => $caselist{posex} ,
		'negex' => $caselist{negex} ,
		'classifier' => $classifier_name,
		%params
	} );
	
	return 0;
}


sub do_CVTEST {
	my $cvfold = shift;
	my @args = @_;

	my %caselist = load_cases_from_args(@args);

	$imgc->cvtest( { 
		'posex' => $caselist{posex} ,
		'negex' => $caselist{negex} ,
		'cv-folds' => $cvfold,
		'classifier' => $classifier_name,
		%params
	} );
	
	return 0;
}
