# ================================================================
#  DROWSINESS LOG VIEWER — view_log.py
#  Reads drowsiness_log.csv and prints a clean summary report
#
#  RUN:
#    python view_log.py
# ================================================================

import csv
import os
from collections import Counter
from datetime import datetime

LOG = 'drowsiness_log.csv'

if not os.path.exists(LOG):
    print(f"❌ Log file '{LOG}' not found. Run detect_advanced.py first.")
    exit()

events = []
with open(LOG, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        events.append(row)

if not events:
    print("📭 No events logged yet.")
    exit()

total      = len(events)
levels     = Counter(e['Level'] for e in events)
reasons    = Counter(e['Reason'] for e in events)

print("\n" + "="*55)
print("   DROWSINESS SESSION LOG REPORT")
print("="*55)
print(f"  Total alert events  : {total}")
print(f"  Level 1 Warnings    : {levels.get('Level 1', 0)}")
print(f"  Level 2 Alarms      : {levels.get('Level 2', 0)}")
print(f"  Level 3 Critical    : {levels.get('Level 3', 0)}")
print()
print("  Alert Triggers:")
for reason, count in reasons.most_common():
    bar = "█" * min(count, 30)
    print(f"    {reason:<20} {bar} ({count})")
print()

# Show last 10 events
print("  Last 10 Events:")
print(f"  {'Time':<22} {'Level':<10} {'Reason':<15} {'EyeR':<8} {'MouthR':<8} {'Tilt'}")
print("  " + "-"*75)
for e in events[-10:]:
    print(f"  {e['Timestamp']:<22} {e['Level']:<10} {e['Reason']:<15} "
          f"{e['EyeRatio']:<8} {e['MouthRatio']:<8} {e['TiltDeg']}°")

print("="*55)
print(f"  Log file: {LOG}")
print("="*55 + "\n")
