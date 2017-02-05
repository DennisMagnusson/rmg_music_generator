#!/usr/bin/perl
use 5.010;

$ARGV[0] or die("No model found");
my $mname = substr $ARGV[0], 7, 99999; #Remove model/
opendir my $dir, $ARGV[1] or die "No directory";
my @files = readdir $dir;
closedir $dir;

system "mkdir generated/$mname_$temp";

for(my $temp = 1.0; $temp <= 1.5; $temp += 0.25) {
	system "mkdir generated/$mname-$temp";
	for(my $i = 0; $i < 10; $i++) {
		my $f = $files[rand @files];
		system "th generator.lua -model $ARGV[0] -temperature $temp -o $mname-$temp/$f.mid -len 500 -start $ARGV[1]/$f";#Generate file
		system "timidity generated/$mname-$temp/$f.mid -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k generated/$mname-$temp/$i.mp3";#Convert to mp3
	}
}
