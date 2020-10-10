cp /dev/null test1.dat
cp /dev/null test2.dat
for i in {0..50}
do
    ./test1 >> "test1.dat"
done
out=`cat "test1.dat" | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'`
echo "test1: $out"

for i in {0..50}
do
    ./test2 >> "test2.dat"
done
out=`cat "test2.dat" | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'`
echo "test2: $out"
