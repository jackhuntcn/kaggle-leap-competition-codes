#!/bin/bash

# train
for year in '0002' '0003' '0004' '0005' '0006' '0007'; do
    for month in '01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12'; do
        python for_kaggle_users.py "${year}" "${month}"
    done
done

for year in '0001'; do
    for month in '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12'; do
        python for_kaggle_users.py "${year}" "${month}"
    done
done

for year in '0008'; do
    for month in '01' '02' '03' '04' '05' '06'; do
        python for_kaggle_users.py "${year}" "${month}"
    done
done

# valid
for year in '0008'; do
    for month in '07' '08' '09' '10' '11' '12'; do
        python for_kaggle_users_valid.py "${year}" "${month}"
    done
done

for year in '0009'; do
    for month in '01'; do
        python for_kaggle_users_valid.py "${year}" "${month}"
    done
done
