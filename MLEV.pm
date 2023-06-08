#!/usr/bin/perl

package MLEV;

use strict;
use warnings;

use Cwd;

use File::Basename;
use lib dirname (__FILE__);
use lib '.';

my $mlev_dir = dirname (__FILE__);
my $cwd = getcwd;

use Common;
use TSV;

our @EXPORT;
our @EXPORT_OK;

##################################################################################
BEGIN {
    use Exporter   ();
    our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);
    $VERSION     = sprintf "%d.%03d", q$Revision: 1.1 $ =~ /(\d+)/g;
    @ISA         = qw(Exporter);
    @EXPORT      = qw(
		&get_mlev_tmpdir
		&auroc
        &mlev_debug
        &mlev_tmpfile
        &mlev_run_R
        &mlev_clean_training_data
        &mlev_clean_test_data
        &mlev_config
        &softmax
        &softmax_hash
	);
    %EXPORT_TAGS = ();
    @EXPORT_OK   = qw(DL_ERROR DL_MESSAGE DL_INFO DL_INFO);
}

our $tmpdir = "/tmp/mlev";
our $tmpstem = "mlev-$$";
our @garbage_files;

mkdir $tmpdir if ! -d $tmpdir;

###################################################################################
sub get_mlev_tmpdir { return $tmpdir; }

#########################################################################################
use constant DL_ERROR    => 0; # displaying error
use constant DL_MESSAGE  => 1; # displaying messages
use constant DL_INFO     => 2; # displaying information
use constant DL_PIPELINE => 3; # displaying pipeline component 
use constant DL_INVOKE   => 4; # displaying program invocation names

sub mlev_debug {
	my $level = shift;
	my @msgs = @_;
	for my $msg (@msgs) {
		chomp $msg;
		for ($level) {
			( $_ == DL_ERROR )    and do { print STDERR "\e[1;31m$msg\e[0m\n"; last; }; 
			( $_ == DL_MESSAGE )  and do { print STDERR "\e[1;32m$msg\e[0m\n"; last; }; 
			( $_ == DL_INFO )     and do { print STDERR "$msg\n"; last; }; 
			( $_ == DL_PIPELINE ) and do { print STDERR "\e[1;33m$msg\e[0m\n"; last; }; 		
			( $_ == DL_INVOKE )   and do { print STDERR "\e[1;37m$msg\e[0m\n"; last; }; 
			print STDERR "$msg\n"; last;
		}
	}
}

###################################################################################
sub auroc(\@\@;\@) {
	my @x = @{ $_[0] };
	my @y = @{ $_[1] };
	my $out_thres = $_[2];
	
	@x = sort { $b <=> $a } grep { $_ ne 'NA' } @x;
	@y = sort { $b <=> $a } grep { $_ ne 'NA' } @y;
	my (@tpr, @fpr); # my @thres;
	my ($Nx, $Ny) = ( scalar(@x), scalar(@y) );
	return 'NA' if $Ny == 0;
	return 'NA' if $Nx == 0;
	my ($tp, $fp) = (0, 0);
	push @tpr, 0; push @fpr, 0; # push @thres, 1;
	push @{$out_thres}, {t=>max(@x, @y), tp=>$tp, fp=>$fp, fn=>$Nx, tn=>$Ny, sens=>0, spec=>(1-0),ppv=>($fp+$tp)?($tp)/($fp+$tp):'NA', npv=>($Ny-$fp)/($Ny-$fp+$Nx-$tp), eauc=>0.5, f1=>0} if defined $out_thres;
	while ( ( scalar(@x) + scalar(@y) ) > 0 ) {
		my $t = undef;
		$t = max( $x[0], $y[0] ) if scalar(@x) and scalar(@y) ;
		$t = $x[0] if scalar(@x) and ! scalar(@y) ;
		$t = $y[0] if ! scalar(@x) and scalar(@y) ;
		while (scalar(@x) and $x[0] >= $t) { shift @x; ++$tp }
		while (scalar(@y) and $y[0] >= $t) { shift @y; ++$fp }
		my $tpr = $tp / $Nx ;
		my $fpr = $fp / $Ny ;
#		push @thres, $t ;
		push @tpr, $tpr;
		push @fpr, $fpr;
		if ( defined $out_thres ) {
			my $tn = $Ny - $fp;
			my $fn = $Nx - $tp;
			my $sens = $tpr;
			my $spec = 1-$fpr;
			my $eauc = ($sens+ $spec)/2; 
			my $ppv = ($tp+$fp)?($tp/($tp+$fp)):'NA';
			my $npv = ($tn+$fn)?($tn/($tn+$fn)):'NA';
			my $f1 = 2*$tp/(2*$tp+$fp+$fn);
			push @{$out_thres}, {t=>$t, tp=>$tp, fp=>$fp, fn=>$Nx-$tp, tn=>$Ny-$fp, sens=>$sens, spec=>$spec, ppv=>$ppv, npv=>$npv, eauc=>$eauc, f1=>$f1} 
		}
	}
	push @tpr, 1; push @fpr, 1; # push @thres, 0;
# 	push @{$out_thres}, {t=>min(@x, @y), tp=>$Nx, fp=>$Ny, fn=>0, tn=>0, sens=>1, spec=>(1-1)}  if defined $out_thres;
	
	my $A = 0.0;

	for (my $i=0; $i<$#tpr; ++$i) {
		my $dFPR = $fpr[$i+1] - $fpr[$i];
		next if $dFPR == 0;
		my $a = 0.5 * ($tpr[$i+1] + $tpr[$i]) * $dFPR ;
		$A += $a;
	}
	return $A;
}


#########################################################################################
our %MLEV_config ;
sub mlev_config {
	my $item = shift;
	if ( (! exists $MLEV_config{'processed'}) and (-f $mlev_dir.'/mlev_config')  ) {
		%MLEV_config = map { my @p = split /=/, $_, 2; s/^\s*|\s*$//g for @p; @p } grep { /=/ } map { chomp; s/\s*#.*//r} file($mlev_dir.'/mlev_config') ;
	}
	$MLEV_config{'processed'} = 1;
	return 
		exists $MLEV_config{$item} ? 
			($MLEV_config{$item} =~ s/\$mlev_dir/$mlev_dir/gr) : 
			undef; 
}


#########################################################################################
sub mlev_tmpfile {
    my $leaf = shift; 
    my $fn_tmp = "$tmpdir/$tmpstem-C$$-$leaf"; # .sprintf("%08d", int(10000000 * rand())) . # C$$-
#     warn "\e[1;31mWARNING: $fn_tmp already exists\e[0m\n" if -f $fn_tmp;
    push @garbage_files, $fn_tmp ;
    return $fn_tmp ;
}

#########################################################################################
sub mlev_run_R {
    my $code = shift;
    open R, "|/usr/bin/R --no-save --slave";
    print R $code;
    close R;
    wait;
}


#########################################################################################
sub softmax {
	my @x = @_;
	my @ex = map { exp($_) } @x;
	my $sum_ex = 0;  $sum_ex += $_ for (@ex);
	my @z = map { ($_ / $sum_ex) } @ex;
	return @z ;
}

#########################################################################################
sub softmax_hash(\%) {
	my %x = %{ $_[0] };
	my @k = keys %x;
	my %ex = map { $_ => exp($x{$_}) } @k ;
	my $sum_ex = 0;  $sum_ex += $ex{$_} for @k ;
	my %z = map { $_ => ($ex{$_} / $sum_ex) } @k ;
	return %z ;
}


#########################################################################################
sub mlev_clean_training_data($) {
    my $fn_training_set = shift; 
    my $TSV_train = TSV->new($fn_training_set);
        
    my $class_label_tr = $TSV_train->guess_class_label;
    my $ftmppath_model  = mlev_tmpfile("R-model.txt");
    my $ftmppath_src_tr = mlev_tmpfile("tr-dummified.tsv");

    $TSV_train->remove_rows_by_criteria( sub { return ${$_[0]}{$class_label_tr} =~ /^(?:NA|\?)$/ } );
    $TSV_train->dummify( $class_label_tr ); # dummify everything execpt the class label
    $TSV_train->save_as( $ftmppath_src_tr );
    
    return ($class_label_tr, $ftmppath_model, $ftmppath_src_tr); # class label, model storage, temporary source file
}

#########################################################################################
sub mlev_clean_test_data($) { # returns cleaned file
    my $fn_test_set = shift;
    my $TSV_test = TSV->new($fn_test_set);
    my $class_label_ts = $TSV_test->guess_class_label();
    my $ftmppath_src_ts = mlev_tmpfile("ts-dummified.tsv");
    
    $TSV_test->remove( $class_label_ts ); 
    $TSV_test->dummify;
    $TSV_test->save_as( $ftmppath_src_ts );
    return $ftmppath_src_ts;
}


#########################################################################################
END {
    for my $fn_tmp (@garbage_files) {
         unlink $fn_tmp if -f $fn_tmp ;
    }
}


# R q|
# library(naivebayes);
# x <- data.frame( V1=1:5, V2=2:6 );
# x$V3 <- (x$V1 + x$V2 > 5);
# 
# nb_model <- naive_bayes(V3 ~ V1 + V2, data=x);
# predict(nb_model, x[c('V2','V1')], type="prob");
# predict(nb_model, data.frame(V1=6,V2=7), type="prob");
# nb_model2 <- nb_model
# save(file='/tmp/nb_model', nb_model2, ascii=TRUE);
# load(file='/tmp/nb_model');
# predict(nb_model2, data.frame(V1=6,V2=7), type="prob");
# 
# 
# print("FNN");
# library(FNN);
# knn(x[c('V1','V2')], data.frame(V1=3, V2=3), cl=x$V3, k = 3, prob = TRUE);
# 
# logit_model <- glm(V3 ~ V1+V2 , family=binomial(link='logit'), data=x);
# print( summary(logit_model) );
# predict( logit_model, data.frame(V1=5, V2=3), type="response" );
# 
# library(e1071);
# svm_model <- svm(as.factor(V3) ~ V1+V2, data = x, cost = 100, gamma = 1, probability=TRUE);
# predict(svm_model, data.frame(V1=1, V2=3), probability=TRUE, type="response");
# |;
# 

1;
