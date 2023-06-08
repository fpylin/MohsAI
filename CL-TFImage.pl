#!/usr/bin/perl

use strict;
use warnings;

use Cwd;

use Storable     qw( freeze thaw );

use File::Basename;
use lib dirname (__FILE__);
my $mlev_dir = dirname (__FILE__);
my $cwd = getcwd;

use Common;
use MLEV;
use SMLModel;
use POSIX;
use Classifier;
use Time::HiRes   qw(usleep nanosleep);
use Digest::SHA1  qw(sha1_hex);

use threads;
use threads::shared;
use Thread::Semaphore;


sub get_file_sha1_b16 {
	my $ctx = Digest::SHA1->new;
	
	open FILE, $_[0];
	$ctx->addfile(*FILE);
	my $digest = $ctx->hexdigest;
	close FILE;

	return $digest;
}


sub run {
    my $cmd = shift;
    printf STDERR "\e[1;37m> $cmd\e[0m\n";
    system ("bash", "-c", $cmd);
    wait;
}

sub run_qx_ordered {
    my $order = shift;
    my $cmd = shift;
    my @output = qx{bash -c '$cmd'};
    @output = grep { !/validated image filenames/ } @output ;
    my $output = join("\n", "$order", join('', @output) );
    $output .= "\n" if ! $output=~ /\n$/s;
    return $output;
}

#########################################################################################

sub train {
    my $ftmppath_model = model_to_keep_after_training ;

    my $imagedir = "/tmp/imagedir-$$";
    run "mkdir $imagedir\n" if ! -d $imagedir;
	
	print STDERR "\e[1;34mTraining set: ".scalar( @{ $TSV_train->{'data'} } )." entries \e[0m\n";
    for my $x ( @{ $TSV_train->{'data'} } ) {
		$$x{'image'} =~ s|/\./|/|g;
		next if ! defined $$x{'class'};
		my $targetdir = "$imagedir/$$x{'class'}";
		mkdir $targetdir if ! -d $targetdir;
		my $sha1sum = get_file_sha1_b16( $$x{'image'} );
		if ( ! -f "$targetdir/$sha1sum.jpg" or ! -l "$targetdir/$sha1sum.jpg" ) {
			print STDERR "ln -s '".$$x{'image'}."' $targetdir/$sha1sum.jpg\n";
			symlink( $$x{'image'}, "$targetdir/$sha1sum.jpg" ); 
		}
    }

    my $cmd_args = '';
    for (@params) {
		/epochs=(\d+)/ and do { $cmd_args .= " -e $1"; next; } ;
		/patience=(\d+)/ and do { $cmd_args .= " -p $1"; next; } ;
		/batch-size=(\d+)/ and do { $cmd_args .= " -b $1"; next; } ;
		/hidden-layers=(.+)/ and do { $cmd_args .= " -l $1"; next; } ;
		/arch=(.+)/ and do { $cmd_args .= " -a $1"; next; } ;
		/trainable/ and do { $cmd_args .= " -t"; next; } ;
    }
    
    my $graph_file = mlev_tmpfile('graph');
    my $label_file = mlev_tmpfile('label');
    
	my $cmd = join("; ", 
		mlev_config('CL-TFImage.tf_gpu.activate'),
		"export TF_CPP_MIN_LOG_LEVEL=2; python3 $mlev_dir/tf/MobileNetV2.py $cmd_args -c $label_file train $graph_file $imagedir | tee /dev/stderr",
		mlev_config('CL-TFImage.tf_gpu.deactivate')
	); 
    
    run $cmd;
    $model->import_file('model', $graph_file);
    $model->set_attr('labels', (map { chomp; trim($_) } file($label_file)) );
    $model->set_attr('params', $cmd);    
    run "rm -r $imagedir" if $imagedir =~ m|^/tmp/imagedir|;
    unlink $graph_file if -f $graph_file;
    unlink $label_file if -f $label_file;
    }


my $nproc = `/usr/bin/nproc` || 1;
#########################################################################################

sub predict {

    my $fn_model = $model->export_model_to_tmpfile ;
    
    my $fn_output = mlev_tmpfile("predictions");

    my $fn_labels = $model->export_file('labels', "$fn_model.labels");
    
    my @labels = $model->get_attr('labels');
    
    chomp for @labels;
    $_ = trim($_) for @labels;
    
    my $N = scalar( @{ $TSV_test->{'data'} } );    

    my @image_list = map { $TSV_test->{'data'}[$_]{'image'} } ( 0 .. ($N-1) ) ;
    
    my @output_all;
    
    my $tmpfn_imagelist = mlev_tmpfile("imagelist");
 
	my $max_threads = max( ceil($nproc * 2 / 3), 1);
	our $total = scalar(@image_list) ;
	our $n_processed = 0;
 	our $batch_size = ( ($total / $max_threads) < ($nproc * 6) ) ?  int( ($total + 1) / ( ( $max_threads - 1 ) || 1)  ) : ($nproc * 4)  ;
	
	our $t_start = time;
	
	my @output_array;

	my $n_active_threads = 0;

	my $order = 0;
	
	
	print STDERR "Testing $total images.\n";
	
	sub report_progress {
		my $order = shift;
		my $retval = shift;
		my @batch_processed = grep { length($_) } (split /\n/, $retval);
		my $t_now = time;
		my $t_elapsed = $t_now - $t_start;
		my $rate_per_sec = $t_elapsed ? ( $n_processed / $t_elapsed ) : 'NA';
		my $rate_str = ( $n_processed * $t_elapsed ) ? sprintf( "%.1f", $rate_per_sec * 60 ) : 'NA';
		my $n_to_process = $total - $n_processed ;
		my $ETA_sec = ( $n_processed and ($rate_per_sec ne 'NA') ) ? ( $n_to_process / $rate_per_sec ) : 'NA';
		my $ETA_min = ( $n_processed and ($rate_per_sec ne 'NA') ) ? ( $ETA_sec > 60 ? int( $ETA_sec / 60 ) : 0 ) : 'NA';
		$ETA_sec -= $ETA_min * 60 if $ETA_sec ne 'NA';
		my $ETA_str = ( $n_processed and ($rate_per_sec ne 'NA') ) ? ($ETA_min > 0 ? sprintf("%d min ", $ETA_min) : '').sprintf("%.0f", $ETA_sec ) : 'NA';
		$n_processed += scalar(@batch_processed);
		
		my $notes = "(Rate: $rate_str images per minute. Batch: $batch_size images)";
		my $f_completed = $n_processed / $total ;
		my $perc_completed = sprintf("%3.0f%%", $f_completed * 100);	
		my $length = 40;
		my $length_completed = int( $f_completed * $length + 0.5) ;

		my $timestr = sprintf("%02d:%02d", int($t_elapsed / 60), ($t_elapsed % 60) );
        print STDERR "$timestr  Completed: $n_processed / $total ($perc_completed) [". join('', map { ($_ < $length_completed ? "=" : "." )} (0..($length-1)))."] ETA: $ETA_str sec  $notes\e[K\r";

	}
	
    while ( scalar(@image_list) ) {
		my @to_process = splice @image_list, 0, $batch_size;
		push @to_process, (splice @image_list, 0) if scalar(@image_list) < scalar(@to_process) / 4 ;
			
		my $tmpfn_imagelist1 = $tmpfn_imagelist.$order;
		open TMPFN, ">$tmpfn_imagelist1";
		print TMPFN map { "$_\n" } @to_process;
		close TMPFN;
		
		my $cmd = join("; ", 
			mlev_config('CL-TFImage.tf_gpu.activate'),
			"export TF_CPP_MIN_LOG_LEVEL=2; python3 $mlev_dir/tf/MobileNetV2.py predict \"$fn_model\" index:$tmpfn_imagelist1; rm  $tmpfn_imagelist1",
			mlev_config('CL-TFImage.tf_gpu.deactivate')
		); 

		$n_active_threads ++;
		my $thr = threads->create( \&run_qx_ordered, $order, $cmd ) ;
		while ( $n_active_threads >=  $max_threads ) {
			for my $thr ( threads->list() ) {
				next if ! $thr -> is_joinable();
				my ($thr_order, $retval) = split /\n/, ( $thr -> join() ), 2;
				$output_array[$thr_order] = $retval;
				-- $n_active_threads;
				report_progress( $thr_order, $retval );
			}
			usleep(100);
		}
		++$order;
	}

	for my $thr ( threads->list() ) {
		my ($thr_order, $retval) = split /\n/, ( $thr -> join() ), 2;
		$output_array[$thr_order] = $retval;
		report_progress( $thr_order, $retval );
	}

    push @output_all, join("\t", @labels);

	for my $row (@output_array) {
		push @output_all, $row ;
	}
	
    open FOUT, ">$fn_output";
	print FOUT map { /\n/ ? "$_" : "$_\n" } @output_all;
    close FOUT;
        
    return $fn_output;
}

set_flag("do_not_rectify_prediction_data", 1);

register(\&train, \&predict);
