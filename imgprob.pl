#!/usr/bin/perl

use strict;
use warnings;

use POSIX;

use Cwd;
use File::Basename;
use lib dirname (__FILE__);
use Common;
use TSV;
my $__FILE_CWD__ = dirname(__FILE__);

use Color::RGB::Util ;

my $cwd = getcwd;

use Image::Magick;

my $f_flat = ( ( grep { /--flat/ } @ARGV ) ? 1 : 0 ); 

@ARGV = grep { ! /--flat/ } @ARGV;

my $filename      = $ARGV[0] ;
my $scorefile     = $ARGV[1] ;
my $output_fn     = $ARGV[2] // '/tmp/sample.jpg';
my $prob_crit     = $ARGV[3] // 0.500 ;
my $prob_positive = $ARGV[4] // 1.000 ; # 0.990 ;
my $mask_fn       = $ARGV[5] ;  # mask file
my $bbox_fn       = $ARGV[6] ;  # mask file

my $usage = "\n\nUsage: $0 image_file score_file [output_filename|None] [prob_crit|0.500] [prob_max|0.990] [mask_fn] [bbox_fn]\n\n\n";

sub get_abspath { my $x = shift; my $ret = ( $x =~ /^\// ? $x : $cwd.'/'.$x ); $x =~ s|//+|/|g; $x =~ s#(?:/\.)+/#/#g; return $ret }

die "ERROR: Image file not specified. $usage" if ( ! defined $filename ) ;

die "ERROR: Image file $filename not readable. $usage" if ( ! -f $filename );

die "ERROR: No score file specified. $usage" if ( ! defined $scorefile );

die "ERROR: No output filename specified. $usage" if ( ! defined $output_fn );

my $f_calib_cache = '/tmp/imgprob-calib-cache.txt';
my %calib_neg_thres ;

sub get_calibrated_negative_threshold($) {
	my $model = shift;
	
	if ( ! scalar(%calib_neg_thres) and -f $f_calib_cache ) {
		%calib_neg_thres = map { chomp; my ($m, $t) = split /\t/, $_; $m => $t } file($f_calib_cache);
	}
	
	if ( ! exists $calib_neg_thres{$model} ) {
		my $lock_fn = ( "/tmp/imgprob-".($model=~ s/[\s\/]+//gr).".lock"  )  ;
		if ( -f $lock_fn ) { 
			while ( -f $lock_fn) { sleep 1; }
			%calib_neg_thres = map { chomp; my ($m, $t) = split /\t/, $_; $m => $t } file($f_calib_cache);
		} else {
			open FLOCK, ">$lock_fn"; close FLOCK;
			print STDERR "$0: File lock: $lock_fn\n";
			my %solid_greys;
			for my $char  ( qw (0 1 2 3 4 5 6 7 8 9 a b c d e f) ) {
				my $colhex = $char x 6;
				my $imgfile = "/tmp/solid-$colhex.jpeg";
				$solid_greys{$colhex} = $imgfile;
				next if -f $imgfile;
				system "convert -size 512x512 xc:#$colhex '$imgfile'";
			}
			my $solid_greys = join(' ', sort values %solid_greys);
			my @solid_negatives = qx{$__FILE_CWD__/imgclassify.pl predict "$model" $solid_greys} ;
			for my $output ( @solid_negatives ) {
				chomp $output ;
				next if $output !~ /Y:(.+?)(?:\t|$)/ ;
				my $prob = $1;
				$calib_neg_thres{$model} = $prob if ! exists $calib_neg_thres{$model}  or $prob > $calib_neg_thres{$model} ;
			}
			
			open CACHEF, ">$f_calib_cache";
			for my $m (sort keys %calib_neg_thres) {
				print CACHEF "$m\t$calib_neg_thres{$m}\n";
			}
			close CACHEF;
			unlink "$lock_fn";
		}
	}
	
	return $calib_neg_thres{$model} ;
}

die "Usage: $0 filename scorefile [output_fn] [prob_crit] [prob_positive]\n" if ! defined $scorefile ;

my %prob ;
my $primary_fn = '';

my @score_lines ;

die "filename eq output_fn\n" if $filename eq $output_fn;

if ( $scorefile ne '-' ) { # defined($scorefile) and 
	@score_lines = file($scorefile) ;
} else {
	die;
	@score_lines = <STDIN>;
}

my %metrics;

for my $line ( @score_lines ) {
	chomp $line;
	my @parts = split /\t/, $line ;
	next if $parts[1] =~ /\.model$/ ;
	my $model = undef;
	$model = shift @parts if $parts[0] =~ /\.model$/ ;
	
	my ($path, $class, @prob) = @parts ;
	my ($primary, $tile) = split /:/, $path;
	$primary = get_abspath($primary);
	$filename = get_abspath($filename);
	next if $primary ne $filename; # !~ /(?:^|\b)\Q$filename\E$/;
	if ( $tile =~ /(median|min|max|vote|infogain)\s*$/ ) {
# 		die $tile;
		$metrics{$1} = join(" ", @prob);
		next;
	}
	
	my ($Y_prob) = map { my ($c, $p) = split /:/,$_ ; $p } grep { /^Y:/ } @prob ;
	my ($tile_X, $tile_Y, $tile_X1, $tile_Y1);
	
	if( $tile =~ /-(\d+)-(\d+)-(\d+)-(\d+).jpe?g$/ ) {
		($tile_X, $tile_Y, $tile_X1, $tile_Y1) = ($1, $2, $3, $4);
	} elsif ( $tile =~ /([\d\.]+)x-(\d+)-(\d+).jpe?g$/ ) {
		my $scale = $1; 
		($tile_X, $tile_Y, $tile_X1, $tile_Y1) = ($2, $3, $2+int(224*$scale)-1, $3+int(224*$scale)-1);
	} elsif ( $tile =~ /-(\d+)-(\d+).jpe?g$/ ) {
		($tile_X, $tile_Y, $tile_X1, $tile_Y1) = ($1, $2, $1+224-1, $2+224-1);
	}
	
	die "$tile" if ! defined $tile_X;
	
	my $Y_prob_adj = $Y_prob;


	my $neg_white_prob; #  = defined($model) ? get_calibrated_negative_threshold($model) : 0;
	$neg_white_prob //= 0;
	$neg_white_prob = 0.99 if $neg_white_prob >= 0.99;
	
	$Y_prob = 0 if $Y_prob eq 'NA' ;
	$Y_prob_adj = ( $Y_prob - $neg_white_prob ) / (1 - $neg_white_prob ) ;
	$Y_prob_adj = 0 if $Y_prob_adj < 0;
	
	$prob{"$tile_X\t$tile_Y\t$tile_X1\t$tile_Y1"} = $Y_prob_adj ;
	print "$tile_X\t$tile_Y\t$tile_X1\t$tile_Y1\t$primary\t$Y_prob\t$Y_prob_adj\n";
	$primary_fn = $primary;
}



if ( defined $bbox_fn ) {
	open FBBOX, ">$bbox_fn" or die "Unable to write the bounding box list to $bbox_fn.";
	print FBBOX map { "$_\n" } sort keys %prob;
	close FBBOX ;
}


exit 1 if ! length $primary_fn;

###################################################################################################################

sub rgb2imrgbf {
	my $col = $_[0];
	my $int = Color::RGB::Util::rgb2int($col);
	my $r = int( $int / 65536 );
	my $g = int( $int / 256 ) % 256;
	my $b = $int % 256;
	return "rgba($r, $g, $b, 1)";
}


sub do_plot_overlay(\%$$$$) {
	my %prob = %{ $_[0] };
	my $primary_fn = $_[1];
	my $output_fn = $_[2];
	my $prob_crit = $_[3];
	my $prob_positive = $_[4];
	
	my $image = Image::Magick->new;
	my $x = $image->Read($primary_fn);
	my $width  = $image->Get('width');
	my $height = $image->Get('height');

	my $overlay = Image::Magick->new;
	warn "$0: $x" if ( $x = $overlay->Set(size=>"${width}x${height}") );
	warn "$0: $x" if ( $x = $overlay->Set(alpha => 'On') );
	warn "$0: $x" if ( $x = $overlay->ReadImage('xc:none') );

	my $hue = 0 ;
	my $v_base = 0.01;
	my $s_base = ( $v_base / 2 ) + 0.5  ;

	$x = $overlay->Draw(strokewidth=>0, primitive=>'rectangle', fill=>rgb2imrgbf( Color::RGB::Util::hsl2rgb("$hue $s_base $v_base") ), points=>"0,0,$width,$height"); # , points=>"$x0,$y0 $x0,$y1 $x1,$y1 $x1,$y0"); # stroke=>'red', 
	warn "$0: $x" if "$x";

	my $prob_a = $prob_positive;
	my $prob_c = $prob_crit;
		
	for my $coord (sort {$prob{$a} <=> $prob{$b}} keys %prob) {
		next if $prob{$coord} < $prob_c ;
		
		my ($y0, $x0, $y1, $x1) = split /\t/, $coord;

		my $p = ($prob_a == $prob_c) ?  1 : ( ($prob{$coord} - $prob_c) / ($prob_a - $prob_c) );
		
		my $v = ( ( $prob{$coord} >= $prob_a) ) ? 1 : POSIX::pow($p, 3) ;
		$v = $v_base if $v < $v_base;
		my $s = ( $v / 2 ) + 0.5 ;
# 		$v /= 2;
		$v = ($v / 4) + 0.25;
		
	
		my $h = $hue + 45 * (1.0 - $p) ;
		
		if ( $f_flat ) {
# 			$v = 0.33333 ;
			$s = 1;
			$h = 0;
		}
		
		my $colhex = rgb2imrgbf( Color::RGB::Util::hsl2rgb("$h $s $v") );
	
		my $col = $colhex;
			warn "$x" if ( $x = $overlay->Draw(strokewidth=>0, primitive=>'rectangle', fill=>$col, points=>"$x0,$y0 $x1,$y1") ); # , points=>"$x0,$y0 $x0,$y1 $x1,$y1 $x1,$y0"); # stroke=>'red', 
	}

	warn "$0: $x" if ( $x = $overlay->Evaluate(channel=>'Alpha', value=>0.45, operator=>'Multiply') );
	warn "$0: $x" if ( $x = $image->Composite(image=>$overlay, geometry=>"+0+0", compose=>"Multiply") );
	
	my $pointsize = 24;
	warn "$0: $x" if ( $x = $image->Annotate( text=>"threshold = ".$prob_c, font=>"Arial", style=>"Normal", pointsize=>$pointsize, fill=>"white", gravity=>"SouthEast", antialias=>"true", x=>($width-1)*0.8, y=>$pointsize, align=>"Left" ) );
	
	warn "$0: $x" if ( $x = $image->Write($output_fn) );
}



sub do_plot_mask(\%$) {
	my %prob = %{ $_[0] };
	my $mask_fn = $_[1];
	my $image = Image::Magick->new;
	my $x = $image->Read($primary_fn);
	my $width  = $image->Get('width');
	my $height = $image->Get('height');

	my $mask = Image::Magick->new;
	warn "$0: $x" if ( $x = $mask->Set(size=>"${width}x${height}") );
	warn "$0: $x" if ( $x = $mask->Set(type=>"Bilevel") );
	warn "$0: $x" if ( $x = $mask->ReadImage('xc:black') );

	my $hue = 0 ;
	my $v_base = 0.01;
	my $s_base = ( $v_base / 2 ) + 0.5  ;

	for my $coord (sort {$prob{$a} <=> $prob{$b}} keys %prob) {
		my $prob_a = $prob_positive;
		my $prob_c = $prob_crit;
		
		next if $prob{$coord} < $prob_c ;
		
		my ($y0, $x0, $y1, $x1) = split /\t/, $coord;
		
		warn "$0: $x" if ( $x = $mask->Draw(strokewidth=>0, primitive=>'rectangle', fill=>'white', points=>"$x0,$y0 $x1,$y1") ); # , points=>"$x0,$y0 $x0,$y1 $x1,$y1 $x1,$y0"); # stroke=>'red', 
	}

	warn "$0: $x" if ( $x = $mask->Write($mask_fn) );
}

if ( defined $output_fn and $output_fn !~ /^None$|^-$/i ) {
	if ( $prob_crit =~ /auto|^-$/i ) {
		my $output_dir = $output_fn;
		$output_dir =~ s/\.jpe?g$//i;
		mkdir $output_dir if ! -d $output_dir ;
		my $basename = ( $output_dir =~ s/.*\///r );
		for ( my $thres = 0.1; $thres < 0.99; $thres += 0.1 ) {
			my $outpath = join("/", $output_dir, $basename."-T".sprintf("%.4f",$thres)).".jpg";
			do_plot_overlay(%prob, $primary_fn, $outpath, $thres, $prob_positive) ;
		}
	} else {
		do_plot_overlay(%prob, $primary_fn, $output_fn, $prob_crit, $prob_positive) ;
	}
}

do_plot_mask(%prob, $mask_fn) if defined $mask_fn;
