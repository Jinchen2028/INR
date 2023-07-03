#!/usr/bin/expect -f
set name wire2d
#mkdir tensorboard/name
#echo $1
set password 1234qwer
spawn scp -r -P 20208 jinchen@101.35.55.30:/home/jinchen/TEST/liif/save/$name/tensorboard ./tb_logger/$name
#scp -r -P 20208 jinchen@101.35.55.30:/home/jinchen/AHU/tb_logger .
expect "password:*" {send "$password\r"}
#tensorboard --logdir tb_logger/$name --port 5500 --bind_all
interact
expect -f ./scp.sh