#!/bin/bash
LOG_FILE="train_$(date +'%Y%m%d_%H%M%S').log"
screen -S train -dm bash -c "python main.py --train > $LOG_FILE 2>&1"