#!/usr/bin/perl

use strict;
use warnings;

use Cwd;
use File::Basename;
use lib dirname (__FILE__);
use Common;
use TSV;
my $__FILE_CWD__ = dirname(__FILE__);
 
 
sub usage {
	print STDERR "FATAL: ".$_[0]."\n\n" if defined $_[0];
	print STDERR "Usage: $0  [-m] [-sScale] model image_orig image_gndt image_annot [split_tile_stride] [image_bbox_trhres] [ndiv|10]\n";
	exit(1);
}

 
 my $multiscale = 0;
 my $scale = 0;
 
 if ( grep { /^-m$/ } @ARGV ) {
	$multiscale = 1;
	@ARGV = grep { ! /^-m$/ } @ARGV ;
 } elsif ( grep { /^-s([\d\.]+)$/ and ($scale = $1) } @ARGV ) {
	@ARGV = grep { ! /^-s(?:[\d\.]+)$/ } @ARGV ;	
 }
 
 my $model              = shift @ARGV // usage("Insufficient arguments") ;
 my $image_orig         = shift @ARGV // usage("Insufficient arguments") ;
 my $image_gndt         = shift @ARGV // usage("Insufficient arguments") ;
 my $image_annot        = shift @ARGV ;
 my $split_tiles_stride = shift(@ARGV) // 0.25;
 my $image_bbox_thresh  = shift(@ARGV) // 0.25;
 my $ndiv = shift(@ARGV) // 40;
 
 
sub run {
	my $cmd = shift;
	print STDERR "\e[1;37m> $cmd\e[0m\n";
	system $cmd;
}

my $tmpfile = "/tmp/imgcfstat-$$.txt";

if ( $multiscale ) {
	unlink $tmpfile if -f $tmpfile;
	run "$__FILE_CWD__/multiscale-predict.pl '$model' '$image_orig' $tmpfile $split_tiles_stride";
} elsif ( $scale ) {
	run "$__FILE_CWD__/scale-predict.pl '$model' '$image_orig' $scale $tmpfile $split_tiles_stride";
} else {
	run "$__FILE_CWD__/imgclassify.pl --split-tiles --split-tiles-stride=$split_tiles_stride predict '$model' '$image_orig' > $tmpfile";
}
my @tistat_output = qx{$__FILE_CWD__/imgtistats.pl '$image_orig' '$image_gndt' $tmpfile $image_bbox_thresh $ndiv};
print map { "|\t$_" } @tistat_output ;
chomp for @tistat_output ;
my @auroc_lines = map { s/^AUROC\t//r } grep { /^AUROC/ } @tistat_output;
my @auprc_lines = map { s/^AUPRC\t//r } grep { /^AUPRC/ } @tistat_output;
my @nauprc_lines = map { s/^nAUPRC\t//r } grep { /^nAUPRC/ } @tistat_output;
@tistat_output = grep { ! /AUROC|AUPRC/ } @tistat_output;

# print "\e[1;31m$_\e[0m\n" for @tistat_output;
@tistat_output = ($tistat_output[0], (grep { ! /^q/ } @tistat_output[1..$#tistat_output]));
# print "\e[1;32m$_\e[0m\n" for @tistat_output;


my $tistat = TSV->new();
$tistat->import_data(@tistat_output);
my %tistat_thres_eauc    = map { $$_{'crit'}.'@'.$$_{'thres'} => $$_{'eauc'} } @{ $tistat->{'data'} };
my %tistat_thres_f1      = map { $$_{'crit'}.'@'.$$_{'thres'} => $$_{'f1'}   } @{ $tistat->{'data'} };
my %tistat_thres_jaccard = map { $$_{'crit'}.'@'.$$_{'thres'} => $$_{'jaccard'} } @{ $tistat->{'data'} };
my $max_thres_eauc = argmax(%tistat_thres_eauc);
my $max_thres_F1   = argmax(%tistat_thres_f1);
my $max_thres_jaccard = argmax(%tistat_thres_jaccard);
my $max_eauc = $tistat_thres_eauc{$max_thres_eauc};
my $max_F1   = $tistat_thres_f1{$max_thres_F1};
my $max_jaccard = $tistat_thres_jaccard{$max_thres_jaccard};

my $max_thres = ($max_thres_eauc=~s/.*@//r);

# print $tistat->to_string();
if ( lc($image_annot) ne 'none' and $image_annot ne '-'  ) {
	if ( $multiscale ) {
		run "$__FILE_CWD__/multiscale-predict.pl '$model' '$image_orig' $tmpfile $split_tiles_stride $max_thres";
	} elsif ( $image_annot =~ /\/$/ ) {
		run "$__FILE_CWD__/imgprob.pl '$image_orig' '$tmpfile' ".($image_annot=~s/\/$//r)." auto 1";
	} else {
		run "$__FILE_CWD__/imgprob.pl '$image_orig' '$tmpfile' '$image_annot' $max_thres 1";
	}
}

# unlink $tmpfile;

print map { "$_\n" } @tistat_output;
print join("\n", 
	"argmax(thres, eAUC)\t". sprintf("%s", $max_thres_eauc), 
	"argmax(thres, F1)\t".   sprintf("%s", $max_thres_F1), 
	"argmax(thres, Jaccard)\t". sprintf("%s", $max_thres_jaccard), 
)."\n";
print join("\n", 
	join("\t", qw(measure max mean median) ),
	join("\t", "eAUC", sprintf("%.5g", $max_eauc) ),
	join("\t", "F1",   sprintf("%.5g", $max_F1) ),
	join("\t", "Jaccard", sprintf("%.5g", $max_jaccard) )
	)."\n";
	
print join("\n", 
	join("\t", "AUROC",  sprintf("%.5g", max(@auroc_lines)),  sprintf("%.5g", mean(@auroc_lines)),  sprintf("%.5g", median(@auroc_lines)) ),
	join("\t", "AUPRC",  sprintf("%.5g", max(@auprc_lines)),  sprintf("%.5g", mean(@auprc_lines)),  sprintf("%.5g", median(@auprc_lines)) ),
	join("\t", "nAUPRC", 
		sprintf("%.5g", max(@nauprc_lines)), 
		sprintf("%.5g", mean(@nauprc_lines)), 
		sprintf("%.5g", median(@nauprc_lines))
	),
)."\n" if scalar(@auroc_lines) > 2;
