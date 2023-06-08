#!/usr/bin/perl
#
# Common.pm:
#   Stores commonly used functions

package Common;

use strict;
use warnings;
use Compress::Zlib;
use POSIX;
use Digest::MD5 qw( md5_hex ) ;
use Cwd;
use Carp;

BEGIN {
	use Exporter   ();
	our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);
    $VERSION     = 1.00;
	$VERSION     = sprintf "%d.%03d", q$Revision: 1.1 $ =~ /(\d+)/g;
    @ISA         = qw(Exporter);
	@EXPORT      = qw(
		&shuffle &sd &quantile &skewness 
		&which &histogram &print_histogram
		&uniq &cut &paste
		&min &max &argmax &argmin &min_at &mean &median 
		&file &files &cat_file &save_file &dir &dirr 
		&file_utf8 &files_utf8
		&gzfile
		&trim &remove_before &remove_after
		&mtime
		&remove_file_if_older_than
		&file_size
		&get_average_coretemp
		&mkdir_r
		&cache_load &cache_save
		&qx_cached
		&zealous_trim
		&union &isect &isect3 &setdiff &venn
		&watchfile
		&file_changed
		&column &print_column
		&column_wrap_text
		&load_tsv_headerless
		);
	%EXPORT_TAGS = ();
	@EXPORT_OK   = qw();
    }

our @EXPORT_OK;

#####################################################################
sub shuffle(\@)
	{
	my $l = shift;
	my $n = scalar(@$l) ;
	
	for(my $z1=0; $z1<$n; ++$z1)
		{
		my $z2 = int(rand( scalar(@$l) ));
		my $s = ${$l}[$z1];
		${$l}[$z1] = ${$l}[$z2];
		${$l}[$z2] = $s;
		}
	return @$l;
	}

#####################################################################
sub max	{
    return undef if (! scalar(@_) );
    
    my $v = undef;
    for (@_) {
	next if $_ eq 'NA';
	$v = $_ if ((! defined $v) || ($_ > $v)) ;
    }
    
    return $v;
}

#####################################################################
sub argmax(\%) {
	my %a = %{ $_[0] }; 
	my $x = undef;
	for (keys %a) {
	    next if $a{$_} eq 'NA';
		$x = $_ if ((! defined $x) || ($a{$_} > $a{$x}));
		}
	return $x;
	}

#####################################################################
sub argmin(\%) {
	my %a = %{ $_[0] }; 
	my $x = undef;
	for (keys %a) {
	    next if $a{$_} eq 'NA';
		$x = $_ if ((! defined $x) || ($a{$_} < $a{$x}));
		}
	return $x;
	}

#####################################################################
sub min_at {
	my @a = @_;
	my $min_i = 0;
	for (my $i=1; $i<=$#a; ++$i) {
		$min_i = $i if ( $a[$min_i] > $a[$i] );
		}
	return $min_i;
	}
#####################################################################
sub min
	{
	return undef if (! scalar(@_) );
	
	my $v = shift;
	for (@_) {
	    next if $_ eq 'NA';
		$v = $_ if ($_ < $v) ;
		}
	
	return $v;
	}

#####################################################################
sub mean
	{
	my @a = grep { defined and !/^NA$/ } @_;
	my $n = scalar(@a);
	my $s = 0.0;
	$s += $_ for (@a);
	return 'NA ' if ! $n;
	return $s / $n;
	}


###################################################################################
sub sd {
	my $total = 0;
	my $m = mean(@_);
	for my $v (@_) {
		my $e = $v - $m;
		$total += $e * $e ;
		}

	return sqrt ($total / (scalar(@_) - 1) );
	}

#####################################################################
sub median {
	my @a = sort {$a <=> $b} grep { $_ !~ /^n\/?a$|^\s*$/i } @_;
	my $n = scalar(@a);
	return undef     if ($n <= 0);
	return $a[0]     if ($n == 1);
	return $a[$#a/2] if ( $n % 2 );
	
	my $i = $n / 2;
	return ($a[$i-1] + $a[$i]) / 2.0;
	}

	

#####################################################################
sub quantile {
	my $q  = shift;
	my @a  = sort {$a <=> $b} grep { defined $_ and $_ ne 'NA' } @_;
	my $n  = scalar(@a);
	my $k2 = int (($q * $n * 2) + 0.5);
	my $k  = int ($k2 / 2);
	return undef     if ($n <= 0);
	push @a, $a[-1];
	push @a, $a[-1];
	return ($k2 % 2) ? $a[$k] : ($a[$k] + $a[$k+1])/2;
	}

#####################################################################
sub skewness
	{
	my $mu = mean (@_);
	my $sd = sd (@_);
	my @x = map { pow ( ($_ - $mu) / $sd, 3) } @_;
	return mean(@x);
	}

#####################################################################
#  sub file($filename): loads a file and return the contents as an array
#     Returns: file contents in an array;
sub file
	{
	my $fn = shift;
	
	open F, "<$fn" or die "Unable to open file $fn. (pwd=".getcwd().")\n";
	my @lines = <F>;
	close F;
	
	return @lines;
	}

#####################################################################
sub files
	{
	my @files = @_;
	
	my @lines;
	
	while( scalar(@files) ) {
		my $fn = shift @files;
		open F, "<$fn" or die "Unable to open file $fn.\n";
		my @lines1 = <F>;
		push @lines, @lines1;
		close F;
		}
	
	return @lines;
	}

#####################################################################
sub file_utf8
	{
	my $fn = shift;
	
	open F, "<$fn" or die "Unable to open file $fn.\n";
	binmode(F, ":utf8");
	my @lines = <F>;
	close F;
	
	return @lines;
	}

#####################################################################
sub files_utf8
	{
	my @files = @_;
	
	my @lines;
	
	while( scalar(@files) ) {
		my $fn = shift @files;
		open F, "<$fn" or die "Unable to open file $fn.\n";
		binmode(F, ":utf8");
		my @lines1 = <F>;
		push @lines, @lines1;
		close F;
		}
	
	return @lines;
	}

sub cat_file { my $fn = shift; return join ('', file($fn) ); }
# #####################################################################
# sub gzfile {
# 	my $path = shift;
# 	my $gzf = gzopen($path, "r");
# 	print STDERR $gzf->gzerror()."\n" if ($gzf->gzerror);
# 
# 	my $data;
# # 	$gzf->gzread($data, 4294967295);
# 	while ( ( ! $gzf->gzerror ) and ( ! $gzf->gzeof() ) ) {
# 		my $buf ;
# # 		$gzf->gzread($buf, file_size($gzf));
# 		$gzf->gzread($buf, file_size($gzf));
# # 		print STDERR $buf;
# 		$data .= $buf;
# 		}
# 	$gzf->gzclose();
# 	
# 	my @lines = map { "$_\n" } split /\n/, $data ;
# 	pop @lines if length($lines[$#lines]) == 0;
# 
# 	return @lines;
# 	}
#####################################################################
sub gzfile {
	my $path = shift;
	my @lines = qx {zcat "$path"};
	return @lines;
	}

#####################################################################
sub save_file # ($\@)
	{
	my $fn = shift;
# 	my $lines_ref = shift;
# 	my @lines = @{ $lines_ref };
	my @lines = @_;
	
	open F, ">$fn" or die "Unable to open file $fn.\n";
	print F join('', @lines);
	close F;
	}


#####################################################################
#  sub dir($filename, [$pattern_in_regex]): lists directory and the entries as an array
#     Returns: list of files contents in an array;
sub dir
	{
	my $dir = shift;
	my $patt = shift;
	opendir DIR, $dir;
	my @files = grep { -f "$dir/$_" } readdir DIR;
	@files = grep {/$patt/} @files if defined $patt;
	closedir DIR;
	return sort @files;
	}

#####################################################################
# get directories recursively
sub dirr
	{
	my $dir = shift;
	my $patt = shift;
	
	opendir DIR, $dir or ( warn "dirr($dir): $!" and return ());
	my @dirent = readdir DIR;
	closedir DIR;

	my @files = grep { -f "$dir/$_" } @dirent ;
	my @dirs  = grep { -d "$dir/$_" } @dirent ;
	@files = grep {/$patt/} @files if defined $patt;
	@files = map { "$dir/$_" } (@files, grep { ! /^\./ } @dirs);
	
	for my $d (@dirs)
		{
		next if ($d =~ /^\.+$/);
		my @subdir_files = dirr("$dir/$d", $patt);
		@files = (@files, @subdir_files);
		}
	return sort @files;
	}

#####################################################################
# sub trim($line):
#    Removes leading and trailing whitespaces
sub trim {
	my $s = shift;
	if (defined $s)
		{
		$s =~ s/^\s*//;
		$s =~ s/\s*$//;
		}
	return $s;
}


#####################################################################
sub remove_before
	{
	my $pattern = shift;
	my @lines = @_;
	shift @lines while $#lines >=0 and $lines[0] !~ /$pattern/;
	return @lines;
	}

#####################################################################
sub remove_after
	{
	my $pattern = shift;
	my @lines = @_;
	pop @lines while $#lines >=0 and $lines[$#lines] !~ /$pattern/;
	return @lines;
	}


#####################################################################
sub which {
	my @y = @_;
	my $n1 = scalar(@y) - 1;
	return grep { $y[$_] } (0 .. $n1) ;
	}
	
#####################################################################
sub histogram {
	my %count;
	for my $x (@_)   {
		$count{$x}++;
	}
	return %count;
}

#####################################################################
sub print_histogram {
	my %count;
	for my $x (@_)   {
		$count{$x}++;
	}
	print "$count{$_}\t$_\n" for (sort { $count{$b} <=> $count{$a} || $a cmp $b } keys %count);
}

#####################################################################
sub uniq {
	my %a;
	$a{$_} = 1 for(@_) ;
	return keys %a;
	}

#####################################################################
sub cut {
	my $ith = shift;
	my @a = @_ ;
	return map { my @b = split /\t/, $_ ; $b[$ith] } @a;
	}

#####################################################################
sub paste {
    my $sep = shift;
    my @args = @_;
    my @out;

    my $i = 0;
    while(1) {
		my $nz = 0;
		for (@args) {
			++$nz if ( $i < scalar(@{$_}) ); 
		}
		last if $nz == 0;
		my @x = map { ( $i < scalar(@{$_}) ) ? ${$_}[$i] : undef } @args;
# 		my $n = grep { defined } @x;
# 		last if ! $n;
		++$i;
		push @out, join($sep, ( map { $_ // '' } @x ) ) ;
    }
    return @out;
}


#####################################################################
sub mtime 
	{
	my ($dev,$ino,$mode,$nlink,$uid,$gid,$rdev,$size,$atime,$mtime,$ctime,$blksize,$blocks) = stat($_[0]);
	return $mtime;
	}

#####################################################################
sub file_size
	{
	my ($dev,$ino,$mode,$nlink,$uid,$gid,$rdev,$size,$atime,$mtime,$ctime,$blksize,$blocks) = stat($_[0]);
	return $size;
	}

#####################################################################
sub get_average_coretemp {
	my $basedir = '/sys/devices/platform';
	opendir DIR, $basedir;
	my $n = 0;
	my $sum = 0;
	for ( grep { /coretemp/ } readdir DIR )
		{
		open FIN, "<$basedir/$_/temp1_input";
		my $coretemp = <FIN>;
		chomp $coretemp ;
		$sum += $coretemp / 1000;
		$n ++;
		close FIN;
		}
	closedir DIR;
	return $sum / $n;
	}

#####################################################################
sub mkdir_r($;$) {
	my $back  = $_[0];
	my $front = $_[1];
	if ( $back =~ s|^(.*?/)|| ) {
		$front .= $1;
		}
	else {
		$front .= $back;
		$back = '';
		}
#       print "!" if -d $front;
#       print "$front\n";
	mkdir $front if ! -d $front; 
	&mkdir_r($back, $front) if length($back) ;
	}


#####################################################################
my $cache_dir = "/tmp/common-cache";

sub cache_load {
	my $id  = shift;
	my $exp = shift;
	my $fn  = $cache_dir."/".md5_hex($id);
	if ( -f $fn ) {
		if ( defined $exp and (mtime($fn) < (time - $exp)) ) {
			unlink $fn ;
			return undef;
			}
		return join '', file($fn);
		}
	return undef;
	}

sub cache_save {
	my $id   = shift;
	my @data = @_;
	my $fn   = $cache_dir."/".md5_hex($id);
	if ( ! -d $cache_dir ) {
		mkdir $cache_dir ;
		chmod 0777, $cache_dir ;
		}
	open F_CACHE, ">$fn" or return 0;
	print F_CACHE @data;
	close F_CACHE;
	return 1;
	}

sub qx_cached {
	my $cmd = shift;
	
	my $data = cache_load($cmd);
	
	return $data if defined $data ;
	
	$data = qx { $cmd };
	 
	cache_save($cmd, $data);
	
	return $data;
	}

sub zealous_trim {
	my @samples = @_;
	while (length $samples[0]) {
		my ($c) = ($samples[0] =~ /^(.)/);
		last if scalar(@samples) != scalar grep { /^\Q$c\E/ } @samples;
		s/^\Q$c\E// for @samples;
		}
	while (length $samples[0]) {
		my ($c) = ($samples[0] =~ /(.)$/);
		last if scalar(@samples) != scalar grep { /\Q$c\E$/ } @samples;
		s/\Q$c\E$// for @samples;
		}
	return @samples;
	}


sub union(\@\@) {
	my @a = @{ $_[0] };
	my @b = @{ $_[1] };
	my %x;
	for (@a, @b) { $x{$_}++; }
	return keys %x;
	}

sub isect(\@\@) {
	my @a = @{ $_[0] };
	my @b = @{ $_[1] };
	my %x;
	@a = uniq(@a);
	@b = uniq(@b);
	for (@a, @b) { $x{$_}++ if defined $_; }
	return grep { $x{$_} > 1 } keys %x;
	}

sub isect3 {
	my @aptr = @_;
	my %x;
	for my $aptr (@aptr) {
		my @a = @{ $aptr } ;
		$x{$_}++ for grep { defined } @a;
		}
	return grep { $x{$_} == scalar(@aptr) } keys %x;
	}

sub setdiff(\@\@) {
	my @a = @{ $_[0] };
	my @b = @{ $_[1] };
	my %b = map { $_ => 1 } @b;
	return grep { ! exists $b{$_} } @a;
	}

sub watchfile(@) {
	my @dirent = @_;
	my %mtimes = map { $_ => mtime($_) } @dirent;
	
	while(1) {
		my @changed ;
		for my $e (@dirent) {
			next if ! defined $mtimes{$e};
			my $m1 = mtime($e);
			push @changed, $e  if ( $m1 != $mtimes{$e} );
			}
		
		return @changed if scalar @changed;
		sleep 1;
		}
	
	return undef;
	}

my %mtime_registry;

sub file_changed {
	my $file = shift;
	if ( ( exists $mtime_registry{$file}) and ( mtime($file) == $mtime_registry{$file} ) ) {
		return 0;
		}
	$mtime_registry{$file} = mtime($file) ;
	return 1;
	}
	
#####################################################################
sub column {
	my @lines = @_;
	my @levels;
	my @width;
	
	my $retval;
	for (@lines) {
		chomp;
		my @parts = split /\t/, $_;
		$levels[$_]{ $parts[$_] }++ for (0 .. $#parts);
		}
		
	sub deansi { my $x = shift; $x =~ s/(\e\[[0-9;]+m)//g ; return $x;}
	
	for my $i (0 .. $#levels) {
# 		$width[$i] = max( map { my $a = $_; $a =~ s/\e\[[0-9;]+m//g; length($a) } keys %{ $levels[$i] } );
		$width[$i] = max( map { my $a = $_; length(deansi($a)) } keys %{ $levels[$i] } );
# 		$width[$i] = max( map { my $a = $_; length($a) } keys %{ $levels[$i] } );
		}
	

sub normANSIcc {
	my $x = shift;
	my $max_width = shift;
	my $append = '';
	my $cc_length = 0;
	
	while ( $x =~ /(\e\[[0-9;]+m)/g ) {
		my $ctrl_code = $1;
		$cc_length += length($ctrl_code);
		}
		
		my $deansied_x = deansi($x); 
		if ( (length($deansied_x) + $cc_length) > $max_width ) {
			$cc_length  = $max_width - length($deansied_x);
			}
	
		$append .= " " x $cc_length if $cc_length > 0;
		return $x.$append;
		}

	sub adj {
		my $width = shift;
		my $str = shift;
		$str = normANSIcc($str, $width);
		$str .= " " x max($width - length(deansi($str)), 0);
		return $str;
		}
	
	for (@lines) {
		my @parts = split /\t/, $_;
		my @output = map { adj($width[$_], $parts[$_]) } (0..$#parts);
		my $line = join("   ", @output);
		$retval .= $line."\n";
		}
	return $retval;
	}


sub print_column {
	my $x = shift;
	our @buffer;
	if ( ! defined $x ) {
		print for column @buffer;
		@buffer=();
	} else {
		push @buffer, $x;
	}
}
	

sub column_wrap_text($@) {
	my $width = shift;
	my $width_minus_5 = $width - 5;
	my @lines = @_;
	
	@lines = map { # Wrapping function
		chomp;
		my @in = split /\t/, $_;
		my @out;
		my $r = 0;
		for my $c ( 0 .. $#in ) {
			my $skip = 0;
			my $x = $in[$c];
			$x =~ s/\s*$//;
	#      		print ">>[$x]\n";
			while ( length($x) >= $width ) {
				my ($a, $b) = ( $x =~ /^(.{$width_minus_5,}?[^[:space:];]+[\s;]*)(.*)/ );
	# 				print "$x -> [$a] [$b]\n";
				if ( $b =~ /^\s*$/ ) { $x = $a; last; }
				$x = $b;
				next if $a =~ /^\s*$/;
				$out[$r+$skip][$c] = $a;
	# 				last if $b =~ /^\s*$/;
				++$skip;
			} 
	# 			print "  [$x]\n";
			$out[$r+$skip][$c] = $x ; # if $x !~ /^\s*$/;
			$r += $skip ; # + 1;
		}
		( map { join("\t", ( map { ($_ // '') } @{$_} ) ) } @out );
		} @lines ;
	return @lines;
}

	
sub load_tsv_headerless {
	my $filename = shift;
	my @field_names = @_;
	my @lines = file($filename);
	my @data;
	
	for (@lines) {
		chomp;
		my @parts = split /\t/, $_;
		my %a = map { $field_names[$_] => $parts[$_] } (0 .. $#field_names);
		push @data, \%a;
		}
	
	return @data;
	}

	
sub venn(\@\@) {
	my @A = @{ $_[0] };
	my @B = @{ $_[1] };
	my %A = map { $_ => 1 } @A;
	my %B = map { $_ => 1 } @B;
	
	my @a  = grep { ! exists $B{$_} } keys %A;
	my @b  = grep { ! exists $A{$_} } keys %B;
	my @ab = grep {   exists $B{$_} } keys %A;
	return (\@a, \@ab, \@b);
}

sub remove_file_if_older_than {
	my $file = shift;
	my @comparisons = @_;
	return undef if ! -f $file;
	my $mtime_file = mtime($file);
	for my $c (@comparisons) {
		next if ! -f $c;
		my $mtime_c = mtime ($c);
		if ( $mtime_file < $mtime_c ) {
			unlink $file ;
			return 1;
		}
	}
	return undef;
}


END { }       # module clean-up code here (global destructor)

1;  # don't forget to return a true value from the file
