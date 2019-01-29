#/usr/bin/sh

for file in "$./*.npy":
do 
  ./simple_plots.py "$file"
done
