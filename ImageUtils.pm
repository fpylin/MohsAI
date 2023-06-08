#!/usr/bin/perl

package ImageUtils;

use strict;
use warnings;

use Image::Magick;
use POSIX;	

our @EXPORT;
our @EXPORT_OK;

use Digest::SHA1  qw(sha1_hex);


##################################################################################
BEGIN {
    use Exporter   ();
    our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);
    $VERSION     = sprintf "%d.%03d", q$Revision: 1.1 $ =~ /(\d+)/g;
    @ISA         = qw(Exporter);
    @EXPORT      = qw(
		&get_file_sha1_b16
		&gen_tiles
	);
    %EXPORT_TAGS = ();
    @EXPORT_OK   = qw();
}


sub get_file_sha1_b16 {
	my $input_file = $_[0];
	my $ctx = Digest::SHA1->new;
	
	return undef if ! -f $input_file;
# 	warn "get_file_sha1_b16(): Reading from ``$input_file''.";
	open FILE, "<$input_file" or warn "get_file_sha1_b16(): Unable to read from ``$input_file''.";
	$ctx->addfile(*FILE);
	my $digest = $ctx->hexdigest;
	close FILE;
	
	my $digest_short = substr($digest, 0, 16);

	return $digest_short ;
}


# generate tiles 

sub gen_tiles { # ($$$;$$$$) { # filename, w, h; stride factor; $output_dir; size_w; size_h
	my $params        = $_[0];
	my $image_fn      = $params->{image_file} ;
	my $tile_w        = $params->{tile_width} ;
	my $tile_h        = $params->{tile_height} ;
	my $stride_factor = $params->{stride_factor} // 1.0 ;
	my $output_dir    = $params->{output_dir} ;
	my $size_w        = $params->{size_w} // $tile_w ; # if tile w < size w then pad the image with black pixels
	my $size_h        = $params->{size_h} // $tile_h ; # if tile w < size w then pad the image with black pixels
	my $scales        = $params->{scales} ;
	my @scales        = defined $scales ? @{ $scales } :  (1.0);
	my $zsuffix = '';
	
# 	my $image_fn  = shift;
# 	my $tile_w = shift;
# 	my $tile_h = shift;
# 	my $stride_factor = shift; $stride_factor = 1.0 if ! defined $stride_factor ;
# 	my $output_dir = shift;
# 	my $size_w = shift() // $tile_w ; # if tile w < size w then pad the image with black pixels
# 	my $size_h = shift() // $tile_h ; # if tile h < size h then pad the image with black pixels
# 	my $zsuffix = '';
	
	$zsuffix = "-${tile_w}x${tile_h}" if ( $tile_w != $size_w || $tile_h != $size_h ) ;
	
	mkdir $output_dir if defined $output_dir;
	
	my $src_image = Image::Magick->new;
	
	my $err ;
	if ( $err = $src_image->Read( $image_fn ) ) {
		warn "gen_tiles: $image_fn: $err" ;
		return () ;
	}
	my $src_img_w = $src_image->Get('width');
	my $src_img_h = $src_image->Get('height');
	
	my $sha1 = get_file_sha1_b16($image_fn) ; 
	
	my $seq = 0;
	
	my @outfiles ;
	
	for my $scale (@scales) {
		my $trg_image = $src_image->Clone;
		
		my $err ;
		
		if ($scale != 1.00) {
			my $err = $trg_image->Scale(width=>($scale * $src_img_w), height=> ($scale * $src_img_h) );
			warn "$err\n" if $err;
		}
	
		my $img_w = $trg_image->Get('width');
		my $img_h = $trg_image->Get('height');
# 		warn "$scale \t $img_w \t $img_h";
	
		my $ntiles_w = ceil( $img_w / $tile_w );
		my $ntiles_h = ceil( $img_h / $tile_h );
		my $img_w_extra = ( $ntiles_w * $tile_w - $img_w );
		my $img_h_extra = ( $ntiles_h * $tile_h - $img_h );
		my $img_w_stride = $stride_factor * ( 1 - ( ( ( $ntiles_w * $tile_w - $img_w ) / ( ($ntiles_w - 1) || 1 ) ) / $tile_w ) ) ;
		my $img_h_stride = $stride_factor * ( 1 - ( ( ( $ntiles_h * $tile_h - $img_h ) / ( ($ntiles_h - 1) || 1 ) ) / $tile_h ) ) ;
	
# 		print STDERR "$image_fn : Image size $img_w x $img_h ($size_w x $size_h). Scale ${scale}x. Extra $img_w_extra x $img_h_extra. N tiles $ntiles_w x $ntiles_h. Stride $img_w_stride x $img_h_stride. \n";
	
		my ($r, $c);
		my ($filename, $ext) = ( $image_fn =~ /(.*)\.(\w+)$/ );
	
		for($r=0; $r * $tile_h <= $img_h ; $r += $img_h_stride) {
	# 		my $y = ( int( ( $r + 1 ) * $tile_h ) >= $img_h) ? ($img_h - $tile_h - 1) : int( $r * $tile_h );
			my $y = int( $r * $tile_h );
			$y = ($img_h - $tile_h - 1) if $y >= $img_h ;
			
			for($c=0; $c * $tile_w <= $img_w ; $c += $img_w_stride) {
	# 			my $x = ( int( ( $c + 1 ) * $tile_w ) >= $img_w) ? ($img_w - $tile_w - 1) : int( $c * $tile_w );
				my $x = int( $c * $tile_w );
				$x = ($img_w - $tile_w - 1) if $x >= $img_w ;
				
				my $orig_y = int( ($y / $scale) );
				my $orig_x = int( ($x / $scale) );
				my $orig_y1 = $orig_y + (int($tile_h / $scale)) - 1 ;
				my $orig_x1 = $orig_x + (int($tile_w / $scale)) - 1 ;
				my $out_fn ;
				
				if ( defined $output_dir ) {
					$out_fn = sprintf("%s/%s$zsuffix-%.3fx-%04d-%04d-%04d-%04d.%s", $output_dir, $sha1, $scale, $orig_y, $orig_x, $orig_y1, $orig_x1, $ext);
				} else {
					$out_fn = sprintf("%s$zsuffix-%.3fx-%04d-%04d-%04d-%04d.%s", $filename, $scale, $orig_y, $orig_x, $orig_y1, $orig_x1, $ext);
				}
				push @outfiles, $out_fn ;
# 				print STDERR "$out_fn\n";
	# 			my $img_filename = s///;
	# 			print STDERR join("\t", $seq, sprintf("%.3f", $c), sprintf("%.3f", $r), $x, $y, $x+$tile_w-1, $y+$tile_h-1, $out_fn)."\n";
				
				if ( ! -f $out_fn ) {
					if ( length($zsuffix) ) {
						my $canvas = Image::Magick->new;
						$canvas->Set(size=>"${size_w}x${size_h}"); 
						$canvas->ReadImage("xc:black");
						warn "$err\n" if ( $err = $canvas->CopyPixels( image=>$trg_image, width=>$tile_w, height=>$tile_h, x=>$x, y=>$y, dx=>(${size_w}-$tile_w)/2, dy=>(${size_h}-$tile_h)/2 ) );
						warn "$err\n" if ( $err = $canvas->Write($out_fn) ) ;
					} else {
						my $image = $trg_image->Clone;
						warn "$err\n" if ( $err = $image->Crop( width=>$tile_w, height=>$tile_h, x=>$x, y=>$y ) );
						warn "$err\n" if ( $err = $image->Write($out_fn) );
						undef $image ;
					}
				}
				
				++ $seq;
				last if $x + $tile_w + 1 >= $img_w;
			}
			last if $y + $tile_h + 1 >= $img_h;
		}	
		undef $trg_image ;
		print STDERR "\e[38;5;142m$image_fn : Image size $img_w x $img_h ($size_w x $size_h). Scale ${scale}x. Extra $img_w_extra x $img_h_extra. N tiles $ntiles_w x $ntiles_h. Stride $img_w_stride x $img_h_stride. (Total=$seq)\e[0m\n";
	}
	
	undef $src_image ;
	
	return @outfiles;
}

END {
}

1;
