#!/bin/bash
# Quick Thesis Backup Script
# Usage: ./quick_backup.sh

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="thesis-backup-$DATE.tar.gz"
BACKUP_DIR="$HOME/thesis-backups"

echo "=================================================="
echo "THESIS BACKUP - Phase 4 Complete"
echo "=================================================="
echo "Date: $DATE"
echo "Output: $BACKUP_DIR/$BACKUP_NAME"
echo ""

# Create backup directory
mkdir -p $BACKUP_DIR

echo "Creating backup (this may take 5-10 minutes)..."
echo "Including EVERYTHING (data/raw, .git, all files)..."
echo ""

# Create compressed backup - NO EXCLUSIONS
tar -czf "$BACKUP_DIR/$BACKUP_NAME" \
  -C $HOME generative-ai-workload-modeling/

if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "BACKUP SUCCESSFUL!"
    echo "=================================================="
    echo "Location: $BACKUP_DIR/$BACKUP_NAME"
    echo "Size: $(du -h "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)"
    echo ""
    echo "Contents:"
    tar -tzf "$BACKUP_DIR/$BACKUP_NAME" | head -20
    echo "... (and more)"
    echo ""
    echo "To restore:"
    echo "  tar -xzf $BACKUP_DIR/$BACKUP_NAME -C ~/"
    echo ""
else
    echo "ERROR: Backup failed!"
    exit 1
fi

# Keep only last 5 backups
cd $BACKUP_DIR
BACKUP_COUNT=$(ls -1 thesis-backup-*.tar.gz 2>/dev/null | wc -l)

if [ $BACKUP_COUNT -gt 5 ]; then
    echo "Cleaning old backups (keeping last 5)..."
    ls -t thesis-backup-*.tar.gz | tail -n +6 | xargs rm
    echo "Done!"
fi

echo ""
echo "All backups in $BACKUP_DIR:"
ls -lht $BACKUP_DIR/thesis-backup-*.tar.gz
echo ""
echo "=================================================="