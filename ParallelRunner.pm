#
# Utility class to spawn a fixed number of child processes usign fork().
# Usage pattern:
#
# use ParallelRunner;
# sub jobToDo { ... } # Function to execute for each job
#
# my $p = new ParallelRunner(4); # 4 parallel processes at a time
# for my $job (@jobs) {
#     $p->run(\&jobToDo,@jobToDoArgs);
# }
# $p->wait(); # Wait for the running jobs to terminate
#
package ParallelRunner;

use strict;
use POSIX;

our $num_cpus; # number of CPUs on this machine 

BEGIN {
	open PROCCPUINFO, "</proc/cpuinfo";
	$num_cpus = grep {/^processor\s*:\s*\d+\s*$/} <PROCCPUINFO>;
	close PROCCPUINFO;
	}


#
# numProcesses: Number of parallell execution threads
# (implemented as child processes)
#
sub new {
	my ($class,$numProcesses) = @_;
	
	$numProcesses = $num_cpus  if (! defined $numProcesses );
	
	my $self = {
		_numProcesses => $numProcesses,	# Number of conc. proc.
		_childPids    => []				# Child proc. PIDs
	};
	print STDERR "ParallelRunner: Using $numProcesses CPUs\n";
	bless($self,$class);
	return $self;
}



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


#
# Start a new child process by calling fctPointer(fctArgs)
#
sub run {
	my ($self,$fctPointer,@fctArgs) = @_;

	# Wait for free execution slot
	if ($self->_numChildPID() >= $self->{_numProcesses}) {
		$self->_waitChild();
	}

# 	while ( (my $temp = get_average_coretemp()) > 65) {
# 		print STDERR "[$$] ".strftime("%F %T", localtime)." CPU too hot ($temp degC), sleeping for 10 seconds\n";
# 		sleep 10;
# 		}

	# Fork
	my $childPID = fork();
	$self->_addChildPID($childPID);
	if ($childPID == 0) {
		# Newly created child; don't forget to exit once done
		exit( &$fctPointer(@fctArgs) ) ;
	} else {
        print STDERR "[$childPID] Started.\n";
	}
}

#
# Call this to wait for running child processes.
#
sub wait {
	my ($self) = @_;
	while ($self->_numChildPID() > 0) {
		$self->_waitChild();
	}
	print STDERR "[$$] All children threads terminated.\n";
}

# Get number of/add/remove PID from list of child PIDs
sub _numChildPID {
	my ($self) = @_;
	return scalar(@{$self->{_childPids}});
}
sub _addChildPID {
	my ($self,$pid) = @_;
	push(@{$self->{_childPids}},$pid)
}
sub _removeChildPID {
	my ($self,$pid) = @_;
	my @newPids = grep { $_ ne $pid } @{$self->{_childPids}};
	$self->{_childPids} = \@newPids;
}

# Wait for a child process to finish
sub _waitChild {
	my ($self) = @_;
	my $finishedPID = CORE::wait();

	# wait() returns -1 if there is no child process left
	if ($finishedPID == -1) {
		$self->{_childPids} = [];
		return;
	}
	print STDERR "[$finishedPID] Finished.\n";
	$self->_removeChildPID($finishedPID);
}


#########################################################################################

1;
