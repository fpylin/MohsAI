#!/usr/bin/perl

package FileGarbageBin;

use strict;
use warnings;

use lib '.';

our $tmpdir = "/tmp";

sub new {
    my ($class) = @_;
    
    my $self = { 'bin' => undef };
    
    my @a;
    $self -> {'bin'} = \@a;
    
    bless($self, $class);
    
    return $self;
}

sub add {
	my $self = shift;
	my @files = @_;
	
	push @{ $self->{'bin'} }, @files;
	return @files ;
}

sub empty {
	my $self = shift;
	for my $file ( grep { /$tmpdir/ and -f $_ } @{ $self->{'bin'} } ) {
		print STDERR "FileGarbageBin->empty(): unlinking $file\n";
		unlink $file ;
	}
}

sub DESTROY {
# 	my $self = shift;
# 	$self->empty();
}

1;
