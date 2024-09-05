#!/bin/bash
jobsubmit -A psoct -p rtx6000 -m 20G -t 7-00:00:00 -c 10 -G 1 -o logs/train-premade.log python3 oct_vesselseg/main.py train-premade
watch -n 1 "squeue -u $USER"
