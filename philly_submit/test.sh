#!/bin/bash
echo $USER":123456"|sudo chpasswd
while [ 1 ]
do
      echo "hello"
      sleep 3600 
done
