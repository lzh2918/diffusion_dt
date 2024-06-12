$i
$j
$k
$d

for ((i=0;i<=1;i++))
do
    for ((j=0;j<=2;j++))
    do
        for ((k=0;k<=1;k++))
        do
            ((d = 3*i + j))
            echo $d
        done
    done
done