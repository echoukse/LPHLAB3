#!/usr/bin/perl

use POSIX;

my @element_counts = ("1",  "32768", "16777216", "67108864");

my %fast_times; 
my %your_times; 
my %serial_times;  

my $perf_points = 1.25;
my %correct;
my $test;

`mkdir -p logs`;
`rm -rf logs/*`;
`mkdir logs/test`;
`mkdir logs/ref`;

if(scalar (@ARGV) != 1)
{
  print ("Usage: ./checker.pl <test>: test = scan, find_repeats\n"); 
  exit(1);   
} else { 
    $test = @ARGV[0];
    print("Test: $test" ); 
}

print "\n";
print ("--------------\n");
print ("Running tests:\n");
print ("--------------\n");

foreach my $element_count (@element_counts) {
    print ("\nElement Count: $element_count\n");
    my @sys_stdout = system ("./scan -m ${test} -i random -n $element_count > ./logs/test/${test}_correctness_${element_count}.log");
    my $return_value  = $?;
    if ($return_value == 0) {
        print ("Correctness passed!\n");
        $correct{$element_count} = 1;
    }
    else {
        print ("Correctness failed\n");
        $correct{$scene} = 0;
    }

    my $your_time = `./scan -m ${test} -i random -n $element_count | tee ./logs/test/${test}_time_${element_count}.log | grep phi_time:`;
    chomp($your_time);
    $your_time =~ s/^[^0-9]*//;
    $your_time =~ s/ ms.*//;
    print ("Your Time: $your_time\n"); 
    
    my $fast_time = `./scan_ref -m ${test} -i random -n $element_count | tee ./logs/ref/${test}_time_${element_count}.log | grep phi_time:`;
    chomp($fast_time);
    $fast_time =~ s/^[^0-9]*//;
    $fast_time =~ s/ ms.*//;
    print ("Reference Time: $fast_time\n"); 

    my $serial_time = `grep CPU_time: ./logs/ref/${test}_time_${element_count}.log`; 
    chomp($serial_time);   
    $serial_time =~ s/^[^0-9]*//;
    $serial_time =~ s/ ms.*//;

    $your_times{$element_count} = $your_time;
    $fast_times{$element_count} = $fast_time;
    $serial_times{$element_count} = $serial_time; 
}

print "\n";
print ("-------------------------\n");
print (ucfirst($test). " Score Table:\n");
print ("-------------------------\n");

my $header = sprintf ("| %-15s | %-15s | %-15s | %-15s | %-15s |\n", "Element Count", "Serial Time", "Fast Time", "Your Time", "Score");
my $dashes = $header;
$dashes =~ s/./-/g;
print $dashes;
print $header;
print $dashes;

my $total_score = 0;

foreach my $element_count (@element_counts){
    my $score;
    my $fast_time = $fast_times{$element_count};
    my $serial_time = $serial_times{$element_count};
    my $time = $your_times{$element_count};

    if ($correct{$element_count}) {
        if ($time <= 1.20 * $fast_time) {
            $score = $perf_points;
        }
        else {
            $score = $perf_points * ($fast_time /$time);
        }
    }
    else {
        $time .= " (F)";
        $score = 0;
    }

    printf ("| %-15s | %-15s | %-15s | %-15s | %-15s |\n", "$element_count", "$serial_time", "$fast_time", "$time", "$score");
    $total_score += $score;
}
print $dashes;
printf ("  %-15s   %-15s   %-15s | %-15s | %-15s |\n", "", "", "", "Total score:", 
    $total_score . "/" . ($perf_points * keys %fast_times));
print $dashes;

printf("Real Deal:\n");
my $MIC_time_1 = `grep "MIC Time" ./logs/test/scan_time_1.log`;
printf("For 1:\n");
print $MIC_time_1;
my $MIC_time_1 = `grep "MIC Time" ./logs/test/scan_time_32768.log`;
printf("For 32768:\n");
print $MIC_time_1;
my $MIC_time_1 = `grep "MIC Time" ./logs/test/scan_time_16777216.log`;
printf("For 16777216:\n");
print $MIC_time_1;
my $MIC_time_1 = `grep "MIC Time" ./logs/test/scan_time_67108864.log`;
printf("For 67108864:\n");
print $MIC_time_1;

    

