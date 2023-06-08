#!/usr/bin/perl

use strict;
use warnings;
use Cwd;
use File::Basename;
use lib dirname (__FILE__);
use Common;
use TSV;
my $__FILE_CWD__ = dirname(__FILE__);
 
use Digest::SHA1;

my $tmpdir = '/tmp/imgclprob';
mkdir $tmpdir if ! -d $tmpdir ;


use Cwd;
my $cwd = getcwd;
my $calibrate = 0;

sub argmax(\%) { my %a = %{ $_[0] }; my $x = undef; for (keys %a) { next if $a{$_} eq 'NA'; $x = $_ if ((! defined $x) || ($a{$_} > $a{$x})); } return $x; }
sub get_abspath { my $x = shift; my $ret = ( $x =~ /^\// ? $x : $cwd.'/'.$x ); $x =~ s|//+|/|g; return $ret }
sub usage {
	print STDERR "FATAL: ".$_[0]."\n\n" if defined $_[0];
	print STDERR "Usage: $0  [-m] [-sScale] fn_model fn_image [stride|0.5]  [fn_output|/tmp/annot.jpg] [prob_crit|0.5]\n";
	exit(1);
}

sub run { 
    my $cmd = $_[0];
    print STDERR "\e[1;37m> $cmd\e[0m\n";
    return qx{$cmd};
}

sub get_file_sha1_b16 {
	my $ctx = Digest::SHA1->new;
	
	open FILE, $_[0];
	$ctx->addfile(*FILE);
	my $digest = $ctx->hexdigest;
	close FILE;

	return $digest;
}

sub get_sha1($$;$) {
	my $model = shift;
	my $image = shift;
	my $split_tiles_stride = shift;
	$model = get_abspath($model);
	my $image_sha1 = get_file_sha1_b16($image);
	$split_tiles_stride //= 0.25;
	return Digest::SHA1::sha1_hex( join("|", $model, $image_sha1, $split_tiles_stride) );
}

my $multiscale = 0;
my $scale = 0;
 
if ( grep { /^-m$/ } @ARGV ) {
	$multiscale = 1;
	@ARGV = grep { ! /^-m$/ } @ARGV ;
} elsif ( grep { /^-s([\d\.]+)$/ and ($scale = $1) } @ARGV ) {
	@ARGV = grep { ! /^-s(?:[\d\.]+)$/ } @ARGV ;	
} elsif ( grep { /^-c$/ and ($calibrate = 1) } @ARGV ) {
	@ARGV = grep { ! /^-c$/ } @ARGV ;
}
 

my $fn_model = shift @ARGV // usage();
my $fn_image = shift @ARGV // usage();
my $split_tiles_stride = shift @ARGV // 0.5;
my $fn_output = shift @ARGV // '/tmp/annot.jpg';
my $prob_crit = shift @ARGV // 0.5;

die "Unable to open model file $fn_model" if ! -f $fn_model;
die "Unable to open image file $fn_image" if ! -f $fn_image;
$fn_output = '/tmp/annot.jpg' if $fn_output eq '-';
die "Invalid output image file name $fn_output" if $fn_output !~ /jpg$/i;


my $sha1_intermed = get_sha1($fn_model, $fn_image, $split_tiles_stride);
my $fn_intermed = "$tmpdir/$sha1_intermed";


if ($calibrate) {
	my @output;
	my %smax;
	for my $calib_scale (80, 90, 100, 110, 120, 130, 140, 150, 160, 175, 190, 200, 210, 220, 230, 240, 250) {
		my $fn_intermed_calib = $fn_intermed . ".$calib_scale" ;
		run "$__FILE_CWD__/scale-predict.pl '$fn_model' '$fn_image' $calib_scale '$fn_intermed_calib' 1";
		my ($total_rows, $y_rows) = (0, 0);
		open my $fh, "<$fn_intermed_calib" or die "Unable to open intermediate file $fn_intermed_calib";
		while ( my $line = <$fh> ) {
			my @cols = split /\t/, $line ;
			$total_rows++;
			$y_rows++ if ($cols[2] eq "Y");
		}
		close $fh;
		next if ! $total_rows;
		my $outline = join("\t", $calib_scale, $total_rows, $y_rows, $smax{$calib_scale} = ( $y_rows / $total_rows ) )."\n";
		push @output, $outline ;
		print "\e[1;35m$outline\e[0m\n";
	}
	print @output; 
	print "Optimal scale\n";
	print argmax(%smax)."\n";
	$scale = argmax(%smax)."\n";
	exit ; #if ! defined $split_tiles_stride;
}



if ( $multiscale ) {
	$fn_intermed .= ".m." ;
} elsif( $scale ) {
	$fn_intermed .= ".$scale" ;
}


if ( ! -f $fn_intermed ) {
    if ( $multiscale ) {
	run "$__FILE_CWD__/multiscale-predict.pl '$fn_model' '$fn_image' '$fn_intermed' $split_tiles_stride";
    } elsif ( $scale ) {
	run "$__FILE_CWD__/scale-predict.pl '$fn_model' '$fn_image' $scale '$fn_intermed' $split_tiles_stride";
    } else {
	my @lines = qx{$__FILE_CWD__/imgclassify.pl --split-tiles --split-tiles-stride=$split_tiles_stride predict '$fn_model' '$fn_image'};
	open FOUT, ">$fn_intermed" or die "Unable to write to file '$fn_intermed'";
	print FOUT @lines;
	close FOUT;
    }
}

my $fn_tmp_bbox = '/tmp/imgprob-bbox.txt';
my $fn_mask = '/tmp/imgprob-mask.png';

run "$__FILE_CWD__/imgprob.pl --flat $fn_image '$fn_intermed' $fn_output $prob_crit 1 '$fn_mask' '$fn_tmp_bbox' ";
