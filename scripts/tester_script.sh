CBMs=(
    'CBM_1'
    # 'CBM_2'
    # 'CBM_3'
    # 'CBM_1'
    # 'CBM_2'
    # 'CBM_3'
    # 'CBM_1'
    # 'CBM_2'
    # 'CBM_3'
)

seeds=( 101 ) # 202 303 404 505 606 707 808 909 )

for i in "${!CBMs[@]}"
do
    echo "${CBMs[$i]} with seed ${seeds[$i]}"
done