#!/bin/bash

# Target Process Name
TARGET="preprocessing.py"
TRAIN_SCRIPT="./run_train_fold0.sh"
LOG_FILE="train_full_$(date +%Y%m%d_%H%M%S).log"

echo "🤖 Auto-Launcher V2 Activated." | tee -a "$LOG_FILE"
echo "Waiting for '$TARGET' to finish..." | tee -a "$LOG_FILE"

# Wait Loop
while pgrep -f "$TARGET" > /dev/null; do
    echo "[$(date)] ... preprocessing is still running. Checking again in 60s." | tee -a "$LOG_FILE"
    sleep 60
done

echo "✅ [$(date)] Preprocessing Finished!" | tee -a "$LOG_FILE"
echo "🚀 Launching Training Sequence..." | tee -a "$LOG_FILE"

# Give it a few seconds to settle
sleep 5

# Execute the training script & Log everything
if [ -f "$TRAIN_SCRIPT" ]; then
    chmod +x "$TRAIN_SCRIPT"
    
    # Run script using unbuffered output (-u) for python inside, and pipe to tee
    # We use 'stdbuf' to unbuffer the shell script output itself if needed
    "$TRAIN_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    
    echo "🎉 [$(date)] All tasks completed. Log saved to $LOG_FILE" | tee -a "$LOG_FILE"
else
    echo "❌ Error: Train script '$TRAIN_SCRIPT' not found!" | tee -a "$LOG_FILE"
fi
