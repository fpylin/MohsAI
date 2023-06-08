#!/usr/bin/perl

use strict;
use warnings;

use Cwd;
use File::Basename;
use lib dirname (__FILE__);
use Common;
use TSV;
my $__FILE_CWD__ = dirname(__FILE__);

use threads;
use threads::shared;
use Storable qw(freeze thaw);

use Cwd;
my $cwd = getcwd;

sub usage {
	print STDERR "FATAL: ".$_[0]."\n\n" if defined $_[0];
	print STDERR "Usage: $0  fn_source_image  fn_image_gndt_label  fn_pred_score_txt  [fn_image_bbox_thresh|0.8] [ndiv|10]\n";
	exit(1);
}

sub run { 
	my $cmd = $_[0];
	print STDERR "\e[1;37m> $cmd\e[0m\n";
	return qx{$cmd};
}

#####################################################################################################
# The generic function of calculating area under the curve with an array of 2*n elements:
# (x_1, y_1, x_2, y_2, x_3, y_3 .... x_n, y_n)

sub calc_auc(@) {
	my @curve = @_;
	my $auc = 0.0;

	for(my $i=2; $i<=$#curve; $i+=2) {
		$auc += ( $curve[$i+1] + $curve[$i-1] ) * ( $curve[$i] - $curve[$i-2] ) / 2.00;
	}
	return $auc;
}

sub get_abspath { my $x = shift; my $ret = ( $x =~ /^\// ? $x : $cwd.'/'.$x ); $x =~ s|//+|/|g; return $ret }

my $fn_source_image      = shift @ARGV or usage("Insufficient arguments");
my $fn_image_gndt_label  = shift @ARGV or usage("Insufficient arguments");
my $fn_pred_score_txt    = shift @ARGV or usage("Insufficient arguments");
my $image_bbox_thresh    = shift @ARGV // 0.8; 
my $ndiv                 = shift @ARGV // 10; 

$fn_source_image = get_abspath($fn_source_image);
$fn_image_gndt_label = get_abspath($fn_image_gndt_label);

my @scores ;
for ( file($fn_pred_score_txt) ) {
# 	print STDERR "|| $_";
	chomp;
	my ($mag_model, $img_part, $pred_class, @pred_scores) ;
	
	if ( /\.model/ ) {
		($mag_model, $img_part, $pred_class, @pred_scores) = split /\t/, $_;
	} else {
		($img_part, $pred_class, @pred_scores) = split /\t/, $_;
	}
	my ($srcf, $tile) = split /:/, $img_part;
	
	$srcf = get_abspath($srcf);
	
# 	print STDERR "|| $tile\t$srcf\t$fn_source_image\n";
	next if $srcf ne $fn_source_image;
	for (@pred_scores) {
		/Y:(.+)/ and do { push @scores, $1; };
	}
}

if (0) {
	@scores = ();
	push @scores, ($_ / 1000) for (0 .. 1000);
	@scores = sort {$a <=> $b} @scores;
}

my %thresholds;
for my $t ( 1 .. ($ndiv-1) ) {
	my $q = ($t / $ndiv);
# 	print STDERR join("\t", $q, quantile($q, @scores), @scores)."\n";
	$thresholds{$t} = quantile($q, @scores) ;
}



my @fields = qw( q thres crit width height pixels tp fn fp tn sens spec ppv npv f1 jaccard acc eauc);


our $process_one_cnt = 0;
sub process_one {
	++$process_one_cnt ;
	my $quantile = shift;
	my $image_bbox_thresh = shift;
	my $threshold = $thresholds{$quantile};
	my $fn_tmp_bbox = "/tmp/imgtistats-$$-$quantile-bbox.txt";
	my $fn_tmp_mask = "/tmp/imgtistats-$$-$quantile-mask.jpg";
# 	print STDERR "$threshold $quantile\n";
	run "$__FILE_CWD__/imgprob.pl '$fn_source_image' '$fn_pred_score_txt' None $threshold 1 '$fn_tmp_mask' '$fn_tmp_bbox' ";
	my @output = run "$__FILE_CWD__/imgconcord.py '$fn_source_image' '$fn_image_gndt_label' '$fn_tmp_mask' '$fn_tmp_bbox' $image_bbox_thresh";
	my %row = map { chomp; my ($f, $v) = split /\t/, $_ } @output ;
	warn "$0: fn_tmp_mask does not exist: $fn_tmp_mask" if ! -f $fn_tmp_mask;
	$row{'q'} = $quantile;
	$row{thres} = $threshold;
	unlink $fn_tmp_bbox;
	unlink $fn_tmp_mask;
	return freeze(\%row);
}


sub get_tistats_all_thres {
	my $image_bbox_thresh = shift;
	my @threads = map { threads->create( \&process_one, $_, $image_bbox_thresh ) } (sort { $b <=> $a } keys %thresholds ) ;

	my @data;

	my @roc_curve = (0, 0);
	my @prc_curve = (0, 1);
	my $npos = undef;
	my $ntot = undef;

	while( scalar(@threads) ) {
		my $seralized = shift(@threads)->join() ;
		my %row = %{ thaw($seralized) };
		$npos = $row{tp} + $row{fn} if ! defined $npos;
		$ntot = $row{tp} + $row{fp} + $row{fn} + $row{tn} if ! defined $ntot;
		if ( $row{spec} =~ /na/i  or $row{sens} =~ /na/i   ) { 
			$row{eauc} = 'NA';
		} else {
			$row{eauc} = ( $row{sens} + $row{spec} ) / 2;
		}
		if ( $row{spec} !~ /na/i  and $row{sens} !~ /na/i   ) {
			push @roc_curve, ( 1 - $row{spec} );
			push @roc_curve, $row{sens};
		}
		if ( $row{ppv} !~ /na/i  and $row{sens} !~ /na/i   ) {
			push @prc_curve, $row{sens};
			push @prc_curve, $row{ppv};
		}
		push @data, \%row;
	}

	push @roc_curve, (1, 1);
	push @prc_curve, (1, 0);
	@prc_curve = reverse @prc_curve;

	print join("\t", @fields)."\n";
	for my $row (@data) {
		print join("\t", ( map {$$row{$_}} @fields) )."\n";
	}

	print "AUROC\t".calc_auc(@roc_curve)."\n";
	print "AUPRC\t".calc_auc(@prc_curve)."\n";
	print "nAUPRC\t".calc_auc(@prc_curve)/( $npos / ($ntot+$npos) )."\n";
}

if ( $image_bbox_thresh eq '-' ) {
	for ( my $t=0.10; $t<=0.90; $t += 0.1 ) {
		get_tistats_all_thres($t) ;
	}
} else {
	get_tistats_all_thres($image_bbox_thresh);
}
