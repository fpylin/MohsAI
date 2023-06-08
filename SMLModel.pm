#!/usr/bin/perl

package SMLModel;

use strict;
use warnings;

use Archive::Tar;
use Carp qw( longmess );

use Cwd;
use File::Basename;
use lib dirname (__FILE__);
use Common;
use TSV;

my $mlev_dir = dirname (__FILE__);
my $cwd = getcwd;

use MLEV;

our @EXPORT;
our @EXPORT_OK;

##################################################################################
BEGIN {
    use Exporter   ();
    our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);
    $VERSION     = sprintf "%d.%03d", q$Revision: 1.1 $ =~ /(\d+)/g;
    @ISA         = qw(Exporter);
    @EXPORT      = qw(
	);
    %EXPORT_TAGS = ();
    @EXPORT_OK   = qw();
}


###############################################################################
sub new {
    my ($class, $filename) = @_;
    my $self = { 'filename' => $filename };
    
    bless($self, $class);
    my $tar = Archive::Tar->new() ;
#     $tar->clear();

    $self->{'tarfile'} = $tar ;
    
    if ( defined($filename) and length($filename) ) {
        if ( -f $filename ) {
            $self->load($filename) ;
        } 
    }
    return $self;
}


sub load{
    my $self = shift;
    my $filename = shift; # model file name - a .tar file
    $self->{'tarfile'}->read($filename) or die "Unable to read TAR file $filename";
}


sub import_file {
    my $self = shift;
    my $file_to_save = shift;
    my $imported_file = shift;
    $self->{'tarfile'} ->remove( $file_to_save )  if $self->{'tarfile'} ->contains_file( $file_to_save ) ;
    
    my $handle   = undef;     # this will be filled in on success
    open($handle, "< :raw :bytes", $imported_file) || die "$0: can't open $imported_file for reading: $!";
    binmode($handle);
    my $imported_file_data = undef;
    my $bytes_read = read $handle, $imported_file_data, file_size($imported_file);
    close $handle;
    print STDERR "Reading $bytes_read bytes from ''$imported_file'' and saving it to ''$file_to_save''\n";
    
    $self->{'tarfile'} ->add_data( $file_to_save, $imported_file_data );
}


sub export_file {
    my $self = shift;
    my $file_to_export = shift;
    my $fn_destination = shift;
    $self->{'tarfile'} ->extract_file( $file_to_export, $fn_destination );
    return $fn_destination;
}


sub set_attr {
    my $self = shift;
    my $attr = shift;
    my @data = @_;
    my $filename = $attr ;
    my $content = join("\n", @data);
    if ( $self->{'tarfile'} ->contains_file( $filename ) ) {
		$self->{'tarfile'} ->replace_content( $filename , $content )
    } else {
		$self->{'tarfile'} ->add_data( $filename, $content );
	}
}

sub get_attr {
    my $self = shift;
    my $attr = shift;
    my $data = $self->{'tarfile'} ->get_content($attr);
    my @lines = split /\n/, $data;
    return @lines ;
}

sub has_attr{ 
    my $self = shift;
    my $attr = shift;
    return $self->{'tarfile'} ->contains_file( $attr ) ;
}

sub import_model_from_file {
    my $self = shift;
    my $model_fn = shift;
    $self->import_file('model', $model_fn );
}

sub export_model_to_file {
    my $self = shift;
    my $model_fn_destination = shift;
    $self->export_file('model', $model_fn_destination );
}

sub export_model_to_tmpfile {
    my $self = shift;
	my $extension = ( defined $_[0] ? $_[0] : 'model' );
    my $fn_model = mlev_tmpfile("ML.$extension");
    $self->export_model_to_file($fn_model);
    return $fn_model ;
}


sub import_submodel_from_file {
    my $self = shift;
    my $submodel_name = shift;
    my $submodel_fn = shift;
    $self->import_file('submodel.'.$submodel_name, $submodel_fn );
}

sub export_submodel_to_file {
    my $self = shift;
    my $submodel_name = shift;
    my $submodel_fn_destination = shift;
    $self->export_file('submodel.'.$submodel_name, $submodel_fn_destination );
}

sub export_submodel_to_tmpfile {
    my $self = shift;
    my $submodel_name = shift;
    my $fn_submodel = mlev_tmpfile("ML-$submodel_name.submodel");
    $self->export_submodel_to_file($submodel_name, $fn_submodel);
    return $fn_submodel ;
}


sub set_header {
    my $self = shift;
    my @data = @_;
    return $self->set_attr('header', @data);
}

sub get_header {
    my $self = shift;
    return $self->get_attr('header');
}

sub set_spec { # model specification
    my $self = shift;
    my @data = @_;
    return $self->set_attr('spec', @data);
}

sub get_spec { # model specification
    my $self = shift;
    return $self->get_attr('spec');
}

sub save_as {
    my $self = shift;
    my $filename = shift;
    $self->{'tarfile'} ->write($filename); 
}

sub save {
    my $self = shift;
#     die " $self->{'filename'} ";
    $self->{'tarfile'} ->write( $self->{'filename'} ) or die "Unable to write file $self->{'filename'}"; 
}

1;
