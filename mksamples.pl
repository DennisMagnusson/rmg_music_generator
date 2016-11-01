#!/usr/bin/perl

use 5.010;

$ARGV[0] or die("No model found");
my $mname = substr $ARGV[0], 7, 99999; #Remove model/

system "mkdir generated/$mname";
for(my $k = 1; $k < 5; $k += 0.5) {
	for(my $temp = 0.5; $temp < 1.5; $temp += 0.25) {
		system "th generator.lua -k $k -model $ARGV[0] -temperature $temp -o $mname/t"."$temp"."_k"."$k".".mid -len 1000";
	}
}
