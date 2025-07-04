#!/usr/bin/env python

import arrow
import random
import os
import sys

from record import Record

if not os.environ.get('RECORD_DIR'):
    sys.exit(
        "This script will corrupt Record's data, please set the RECORD_DIR "
        "environment variable to safely use it for development purpose."
    )

record = Record(config_dir=os.environ.get('RECORD_DIR'),
                frames=None,
                current=None)

projects = [
    ("apollo11", ["reactor", "module", "wheels", "steering", "brakes"]),
    ("hubble", ["lens", "camera", "transmission"]),
    ("voyager1", ["probe", "generators", "sensors", "antenna"]),
    ("voyager2", ["probe", "generators", "sensors", "antenna"]),
]

now = arrow.now()

for date in arrow.Arrow.range('day', now.shift(months=-1), now):
    if date.weekday() in (5, 6):
        # Weekend \o/
        continue

    start = date.replace(hour=9, minute=random.randint(0, 59)) \
                .shift(seconds=random.randint(0, 59))

    while start.hour < random.randint(16, 19):
        project, tags = random.choice(projects)
        frame = record.frames.add(
            project,
            start,
            start.shift(seconds=random.randint(60, 4 * 60 * 60)),
            tags=random.sample(tags, random.randint(0, len(tags)))
        )
        start = frame.stop.shift(seconds=random.randint(0, 1 * 60 * 60))

record.save()
