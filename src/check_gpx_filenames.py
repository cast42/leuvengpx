import re
from pathlib import Path

regex = r"^DR\s[N|E|W|C|S]{1,2}\s[\w\s]+\.gpx$"

test_str = "DR NE Vlooibergtoren.gpx"

for gpxfile in Path("data/gpx/").glob("*.gpx"):
    test_str = gpxfile.name

    matches = re.finditer(regex, test_str, re.MULTILINE)

    for matchNum, match in enumerate(matches, start=1):
        print(
            "Match {matchNum} was found at {start}-{end}: {match}".format(
                matchNum=matchNum,
                start=match.start(),
                end=match.end(),
                match=match.group(),
            )
        )

        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1

            print(
                "Group {groupNum} found at {start}-{end}: {group}".format(
                    groupNum=groupNum,
                    start=match.start(groupNum),
                    end=match.end(groupNum),
                    group=match.group(groupNum),
                )
            )
