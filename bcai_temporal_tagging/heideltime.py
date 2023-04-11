""" Our python-version of HeidelTime created in the context of multilingual temporal tagging (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH
@author: Lukas Lange

This file contains modified code from the HeidelTime library
- https://github.com/HeidelTime/heideltime/blob/master/src/de/unihd/dbs/uima/annotator/heideltime/HeidelTime.java
Last modification date: 2023-03-29

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import math
import re
from datetime import datetime

from dateutil.relativedelta import relativedelta


class MyLogger:
    LEVEL_ERROR = 0
    LEVEL_INFO = 1
    LEVEL_DEBUG = 2
    LEVEL_DETAIL = 3

    def __init__(self, level=0):
        self.level = level

    def printError(self, msg):
        if self.level >= self.LEVEL_ERROR:
            print(msg)

    def printDetail(self, msg):
        if self.level >= self.LEVEL_DETAIL:
            print(msg)


Logger = MyLogger(MyLogger.LEVEL_ERROR)


class DateCalculator:
    @staticmethod
    def formatDate(dDate):
        s = str(dDate.year).zfill(4) + "-"
        s += str(dDate.month).zfill(2) + "-"
        s += str(dDate.day).zfill(2)
        return s

    @staticmethod
    def parseDatetime(date):
        is_bc = "BC" in date

        if is_bc:
            if re.match("BC\d\d\d\d-\d\d-\d\d", date):
                dDate = datetime.strptime(date.replace("BC", ""), "%Y-%m-%d")
            elif re.match("BC\d\d\d\d-\d\d", date):
                dDate = datetime.strptime(date.replace("BC", ""), "%Y-%m")
            elif re.match("BC\d\d\d\d", date):
                dDate = datetime.strptime(date.replace("BC", ""), "%Y")
            else:
                dDate = datetime.strptime(
                    date.replace("BC", "")[::-1].zfill(4)[::-1], "%Y"
                )

        elif re.match("\d\d\d\d-W\d\d", date):
            year, week = date.split("-W")
            dDate = datetime.fromisocalendar(int(year), int(week), 1)

        else:
            if re.match("\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d", date):
                dDate = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
            elif re.match("\d\d\d\d-\d\d-\d\dT\d\d:\d\d", date):
                dDate = datetime.strptime(date, "%Y-%m-%dT%H:%M")
            elif re.match("\d\d\d\d-\d\d-\d\dT\d\d", date):
                dDate = datetime.strptime(date, "%Y-%m-%dT%H")
            elif re.match("\d\d\d\d-\d\d-\d\d", date):
                dDate = datetime.strptime(date, "%Y-%m-%d")
            elif re.match("\d\d\d\d-\d\d", date):
                dDate = datetime.strptime(date, "%Y-%m")
            elif re.match("\d\d\d\d", date):
                dDate = datetime.strptime(date, "%Y")
            else:
                dDate = datetime.strptime(date[::-1].zfill(4)[::-1], "%Y")

        return is_bc, dDate

    @staticmethod
    def getXNextYear(date: str, x: int):
        is_bc, dDate = DateCalculator.parseDatetime(date)

        if is_bc:
            newYear = dDate.year - x
            if newYear <= 0:
                is_bc = False
                newYear = abs(newYear)
        else:
            newYear = dDate.year + x

        newYear = str(newYear).zfill(4)
        return "BC" + newYear if is_bc else newYear

    @staticmethod
    def getXNextDecade(date: str, x: int):
        is_bc, dDate = DateCalculator.parseDatetime(date)

        if is_bc:
            newYear = dDate.year - x * 10
            if newYear <= 0:
                is_bc = False
                newYear = abs(newYear)
        else:
            newYear = dDate.year + x * 10

        newDecade = str(newYear).zfill(4)[:3]
        return "BC" + newDecade if is_bc else newDecade

    @staticmethod
    def getXNextCentury(date: str, x: int):
        is_bc, dDate = DateCalculator.parseDatetime(date)

        if is_bc:
            newYear = dDate.year - x * 100
            if newYear <= 0:
                is_bc = False
                newYear = abs(newYear)
        else:
            newYear = dDate.year + x * 100

        newCentury = str(newYear).zfill(4)[:2]
        return "BC" + newCentury if is_bc else newCentury

    @staticmethod
    def getXNextDay(date: str, x: int):
        is_bc, dDate = DateCalculator.parseDatetime(date)
        assert not is_bc
        newDate = dDate + relativedelta(days=x)
        return DateCalculator.formatDate(newDate)

    @staticmethod
    def getXNextMonth(date: str, x: int):
        is_bc, dDate = DateCalculator.parseDatetime(date)
        assert not is_bc  # supported by HeidelTime
        newDate = dDate + relativedelta(months=x)
        return DateCalculator.formatDate(newDate)[0:7]

    @staticmethod
    def getXNextWeek(date: str, x: int):
        is_bc, dDate = DateCalculator.parseDatetime(date)
        assert not is_bc  # supported by HeidelTime
        newDate = dDate + relativedelta(weeks=x)
        return str(newDate.year).zfill(4) + "-W" + str(newDate.isocalendar()[1])

    @staticmethod
    def getWeekdayOfDate(date: str):
        is_bc, dDate = DateCalculator.parseDatetime(date)
        assert not is_bc
        # return normalization["reWeekday"][str(dDate.weekday())]
        weekday = dDate.isoweekday()
        if weekday == 7:  # sunday:
            weekday = 0
        return weekday + 1  # Monday := 2; Tuesday := 3, ...

    @staticmethod
    def getWeekOfDate(date: str):
        is_bc, dDate = DateCalculator.parseDatetime(date)
        assert not is_bc
        return dDate.isocalendar()[1]


class ContextAnalyzer:
    @staticmethod
    def getLastMentionedX(linearDates, x, Language, normalization):
        """The value of the x of the last mentioned Timex is calculated.
        * @param linearDates list of previous linear dates
        * @param i index for the previous date entry
        * @param x type to search for
        * @return last mentioned entry"""

        for value in reversed(linearDates):

            if x == "century":
                if re.match("^\d\d.*", value):
                    return value[0:2]
                if re.match("^BC\d\d.*", value):
                    return value[0:4]

            elif x == "decade":
                if re.match("^\d\d\d.*", value):
                    return value[0:3]
                if re.match("^BC\d\d\d.*", value):
                    return value[0:5]

            elif x == "year":
                if re.match("^\d\d\d\d.*", value):
                    return value[0:4]
                if re.match("^BC\d\d\d\d.*", value):
                    return value[0:6]

            elif x == "dateYear":
                if re.match("^\d\d\d\d.*", value):
                    return value
                if re.match("^BC\d\d\d\d.*", value):
                    return value

            elif x == "month":
                if re.match("^\d\d\d\d-\d\d.*", value):
                    return value[0:7]
                if re.match("^BC\d\d\d\d-\d\d.*", value):
                    return value[0:9]

            elif x == "month-with-details":
                if re.match("^\d\d\d\d-\d\d.*", value):
                    return value
                # if re.match('^BC\d\d\d\d-\d\d.*', value):
                #    return value

            elif x == "day":
                if re.match("^\d\d\d\d-\d\d-\d\d.*", value):
                    return value[0:10]
                # if re.match('^BC\d\d\d\d-\d\d-\d\d.*', value):
                #    return value[0:12]

            elif x == "week":
                if re.match("^\d\d\d\d-\d\d-\d\d.*", value):
                    return DateCalculator.getXNextWeek(value[0:10], 1)
                if re.match("^\d\d\d\d-W\d\d.*", value):
                    return value[0:8]

            elif x == "quarter":
                if re.match("^\d\d\d\d-\d\d.*", value):
                    t = normalization["reMonthInQuarter"][value[5:7]]
                    return value[0:4] + "-Q" + str(t)
                if re.match("^\d\d\d\d-Q[1234].*", value):
                    return value[0:7]

            elif x == "dateQuarter":
                if re.match("^\d\d\d\d-Q[1234].*", value):
                    return value[0:7]

            elif x == "season":
                if re.match("^\d\d\d\d-\d\d.*", value):
                    t = normalization["reMonthInSeason"][value[5:7]]
                    return value[0:4] + "-" + t
                if re.match("^\d\d\d\d-(SP|SU|FA|WI).*"):
                    return value[0:7]

        return ""

    @staticmethod
    def getLastTense(context, language, spacy_model=None):
        if spacy_model is None:
            return ""

        doc = spacy_model(context)

        # MD  Modal verb (can, could, may, must)
        # VB  Base verb (take)
        # VBC Future tense, conditional
        # VBD Past tense (took)
        # VBF Future tense
        # VBG Gerund, present participle (taking)
        # VBN Past participle (taken)
        # VBP Present tense (take)
        # VBZ Present 3rd person singular (takes)

        last_tenses = [""]
        for token in doc:
            tense = token.morph.get("Tense")
            form = token.morph.get("VerbForm")
            tense = tense[0] if len(tense) > 0 else ""
            form = form[0] if len(form) > 0 else ""

            # Standard verbs
            if token.pos_ == "VERB" and tense == "Past":
                last_tenses.append("PAST")
            if token.pos_ == "VERB" and form != "Part" and tense == "Pres":
                last_tenses.append("PRESENT")

            # Modal and auxiliary verbs
            elif token.pos_ == "AUX" and form == "Fin" and tense == "Pres":
                last_tenses.append("FUTURE")
            elif token.pos_ == "AUX" and form == "Fin" and tense == "Past":
                last_tenses.append("PAST")
            elif token.pos_ == "AUX" and form == "Fin":
                last_tenses.append("PRESENTFUTURE")

        return last_tenses[-1]


class HolidayProcessor:
    @staticmethod
    def evalute_timex(value):
        cmd_p = "((\w\w\w\w)-(\w\w)-(\w\w))\s+funcDateCalc\((\w+)\((.+)\)\)"
        year_p = "(\d\d\d\d)"
        date_p = "(\d\d\d\d)-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])"

        valueNew = value

        cmd_m = re.match(cmd_p, value)
        if cmd_m:
            Logger.printDetail("[HolidayProcessor] found holiday timex:" + value)
            date = cmd_m.group(1)
            year = cmd_m.group(2)
            month = cmd_m.group(3)
            day = cmd_m.group(4)
            function = cmd_m.group(5)
            args = re.split("\s*,\s*", cmd_m.group(6))

            for j in range(len(args)):
                args[j] = args[j].replace("DATE", date)
                args[j] = args[j].replace("YEAR", year)
                args[j] = args[j].replace("MONTH", month)
                args[j] = args[j].replace("DAY", day)

            if function == "EasterSunday":
                year_m = re.match(year_p, args[0])
                if year_m:
                    if len(args) == 1:
                        args.append(0)
                    valueNew = HolidayProcessor.getEasterSunday(
                        int(args[0][0:4]), int(args[1])
                    )
                else:
                    Logger.printError(
                        "[HolidayProcessor] invalid timex (EasterSunday):" + value
                    )
                    valueNew = "XXXX-XX-XX"

            elif function == "WeekdayRelativeTo":
                year_m = re.match(year_p, args[0])
                if year_m:
                    valueNew = HolidayProcessor.getWeekdayRelativeTo(
                        args[0], int(args[1]), int(args[2]), bool(args[3])
                    )
                else:
                    Logger.printError(
                        "[HolidayProcessor] invalid timex (WeekdayRelativeTo):" + value
                    )
                    valueNew = "XXXX-XX-XX"

            elif function == "EasterSundayOrthodox":
                year_m = re.match(year_p, args[0])
                if year_m:
                    if len(args) == 1:
                        args.append(0)
                    valueNew = HolidayProcessor.getEasterSundayOrthodox(
                        int(args[0][0:4]), int(args[1])
                    )
                else:
                    Logger.printError(
                        "[HolidayProcessor] invalid timex (EasterSundayOrthodox):"
                        + value
                    )
                    valueNew = "XXXX-XX-XX"

            elif function == "ShroveTideOrthodox":
                year_m = re.match(year_p, args[0])
                if year_m:
                    valueNew = HolidayProcessor.getShroveTideOrthodox(int(args[0]))
                else:
                    Logger.printError(
                        "[HolidayProcessor] invalid timex (ShroveTideOrthodox):" + value
                    )
                    valueNew = "XXXX-XX-XX"

            else:
                Logger.printError("[HolidayProcessor] command not found:" + function)
                valueNew = "XXXX-XX-XX"

            Logger.printDetail("[HolidayProcessor] resolved to:" + valueNew)
        return valueNew

    @staticmethod
    def getEasterSunday(year: int, days: int):
        """Get the date of a day relative to Easter Sunday in a given year. Algorithm used is from the "Physikalisch-Technische Bundesanstalt Braunschweig" PTB.

        @author Hans-Peter Pfeiffer
        @param year
        @param days
        @return date
        """
        # K = year / 100
        # M = 15 + ( ( 3 * K + 3 ) / 4 ) - ( ( 8 * K + 13 ) / 25 )
        # S = 2 - ( ( 3 * K + 3 ) / 4 )
        # A = year % 19
        # D = ( 19 * A + M ) % 30
        # R = ( D / 29 ) + ( ( D / 28 ) - ( D / 29 ) * ( A / 11 ) )
        # OG = 21 + D - R
        # SZ = 7 - ( year + ( year / 4 ) + S ) % 7
        # OE = 7 - ( OG - SZ ) % 7
        # OS = int(OG + OE)
        #
        # if OS <= 31:
        #     date = str(int(year)).zfill(4) + '-03-' + str(OS).zfill(2)
        # else:
        #     date = str(int(year)).zfill(4) + '-04-' + str(OS-31).zfill(2)

        if year > 1582:
            # https://www.linuxtopia.org/online_books/programming_books/python_programming/python_ch38.html
            # Procedure 38.5. Algorithm O.
            # (Date of Easter after 1582.) Let Y be the year for which the date of Easter is desired.
            # A ← ( Y mod 19).
            # B ← ⌊ Y / 100⌋.
            # C ← ( Y mod 100).
            # D ← ⌊ B / 4 ⌋.
            # E ← B mod 4.
            # G ← ⌊ (8 B + 13 ) / 25 ⌋.
            # H ← ( ( 19 A ) + B − D − G + 15 ) mod 30.
            # M ← ⌊ ( A + 11 H ) / 319 ⌋.
            # I ← ⌊ C / 4 ⌋.
            # K ← C mod 4.
            # F ← ( 2 E + 2 I − K − H + M + 32 ) mod 7.
            # N ← ⌊ ( H − M + F + 90 ) / 25 ⌋.
            # P ← ( H − M + F + N + 19 ) mod 32.
            # Easter is day P of month N .
            A = year % 19
            B = math.floor(year / 100)
            C = year % 100
            D = math.floor(B / 4)
            E = B % 4
            G = math.floor((8 * B + 13) / 25)
            H = ((19 * A) + B - D - G + 15) % 30
            M = math.floor((A + 11 * H) / 319)
            I = math.floor(C / 4)
            K = C % 4
            F = (2 * E + 2 * I - K - H - M + 32) % 7
            N = math.floor((H - M + F + 90) / 25)
            P = (H - M + F + N + 19) % 32

            date = (
                str(int(year)).zfill(4)
                + "-"
                + str(int(N)).zfill(2)
                + "-"
                + str(int(P)).zfill(2)
            )

        else:
            # https://www.linuxtopia.org/online_books/programming_books/python_programming/python_ch38.html
            # Procedure 38.4. Algorithm B.
            # (Date of Easter prior to 1583.) Let Y be the year for which the date of Easter is desired.
            # A ← ( Y mod 4).
            # B ← ( Y mod 7).
            # C ← ( Y mod 19).
            # D ← (19 C + 15) mod 30.
            # E ← (2 A + 4 B − D + 34) mod 7.
            # H ← D + E + 114.
            # F ← ⌊ H / 31 ⌋.
            # G ← H mod 31.
            # Easter is day G +1 of month F
            A = year % 4
            B = year % 7
            C = year % 19
            D = (19 * C + 15) % 30
            E = (2 * A + 4 * B - D + 34) % 7
            H = D + E + 114
            F = math.floor(H / 31)
            G = H % 31

            date = (
                str(int(year)).zfill(4)
                + "-"
                + str(int(F)).zfill(2)
                + "-"
                + str(int(G + 1)).zfill(2)
            )

        dDate = datetime.strptime(date, "%Y-%m-%d")
        tDate = dDate + relativedelta(days=days)
        return DateCalculator.formatDate(tDate)

    @staticmethod
    def getEasterSundayOrthodox(year: int, days: int):
        """Get the date of a day relative to Easter Sunday in a given year. Algorithm used is from the http://en.wikipedia.org/wiki/Computus#cite_note-otheralgs-47.

        @author Elena Klyachko
        @param year
        @param days
        @return date
        """
        A = year % 4
        B = year % 7
        C = year % 19
        D = (19 * C + 15) % 30
        E = ((2 * A + 4 * B - D + 34)) % 7
        Month = int(math.floor((D + E + 114) / 31))
        Day = int(((D + E + 114) % 31) + 1)

        date = (
            str(int(year)).zfill(4)
            + "-"
            + str(int(Month)).zfill(2)
            + "-"
            + str(int(Day)).zfill(2)
        )

        dDate = datetime.strptime(date, "%Y-%m-%d")
        tDate = dDate + relativedelta(days=days)
        tDate = tDate + relativedelta(days=HolidayProcessor.getJulianDifference(year))
        return DateCalculator.formatDate(tDate)

    @staticmethod
    def getShroveTideWeekOrthodox(year: int):
        """Get the date of the Shrove-Tide week in a given year.

        @author Elena Klyachko
        @param year
        @param days
        @return date
        """
        easterOrthodox = HolidayProcessor.getEasterSundayOrthodox(year, 0)
        easterDate = datetime.strptime(easterOrthodox, "%Y-%m-%d")

        shroveTideWeek = easterDate + relativedelta(days=-49)
        _, week, _ = shroveTideWeek.isocalendar()

        date = str(year).zfill(4) + "-W" + str(week).zfill(2)
        return date

    @staticmethod
    def getJulianDifference(year: int):
        # TODO: this is not entirely correct!
        century = year / 100 + 1

        if century < 18:
            return 10
        if century == 18:
            return 11
        if century == 19:
            return 12
        if century == 20 or century == 21:
            return 13
        if century == 22:
            return 14
        return 15

    @staticmethod
    def getWeekdayRelativeTo(date: str, weekday: int, number: int, count_itself: bool):
        dDate = datetime.strptime(date, "%Y-%m-%d")

        if number == 0:
            return DateCalculator.formatDate(dDate)

        else:
            if number < 0:
                number += 1

            day = DateCalculator.getWeekdayOfDate(date)

            if (count_itself and number > 0) or (not count_itself and number <= 0):
                if day <= weekday:
                    add = weekday - day
                else:
                    add = weekday - day + 7
            else:
                if day < weekday:
                    add = weekday - day
                else:
                    add = weekday - day + 7
            add = add + int((number - 1) * 7)
            print(dDate)
            print(f"days to add: {add}")
            dDate = dDate + relativedelta(days=add)

            return DateCalculator.formatDate(dDate)


def resolve_ambigious_string(
    ambigString,
    normalization,
    dct=None,
    context="",
    language="en",
    documentType="news",
    linearDates=None,
    spacyModel=None,
):
    if linearDates is None:
        linearDates = []  # empty list of preceedings Timex tags

    documentTypeNews = documentType == "news"
    documentTypeNarrative = documentType == "narrative"
    documentTypeColloquial = documentType == "colloquial"
    documentTypeScientific = documentType == "scientific"

    dctAvailable = False
    dctValue = ""
    dctCentury = 0
    dctYear = 0
    dctDecade = 0
    dctMonth = 0
    dctDay = 0
    dctSeason = ""
    dctQuarter = ""
    dctHalf = ""
    dctWeekday = 0
    dctWeek = 0

    # ////////////////////////////////////////////
    #  INFORMATION ABOUT DOCUMENT CREATION TIME //
    # ////////////////////////////////////////////

    if dct is not None and dct != "XXXX-XX-XX":
        dctAvailable = True
        assert re.match("\d\d\d\d-\d\d-\d\d", dct)

        dctCentury = int(dct[0:2])
        dctYear = int(dct[0:4])
        dctDecade = int(dct[2:3])
        dctMonth = int(dct[5:7])
        dctDay = int(dct[8:10])

        dctQuarter = "Q" + normalization["reMonthInQuarter"][str(dctMonth).zfill(2)]
        dctHalf = "H1" if dctMonth <= 6 else "H2"

        dctSeason = normalization["reMonthInSeason"][str(dctMonth).zfill(2)]

        dctWeekday = DateCalculator.getWeekdayOfDate(dct)
        dctWeek = DateCalculator.getWeekOfDate(dct)

        Logger.printDetail(f"dctCentury:{dctCentury}")
        Logger.printDetail(f"dctYear:{dctYear}")
        Logger.printDetail(f"dctDecade:{dctDecade}")
        Logger.printDetail(f"dctMonth:{dctMonth}")
        Logger.printDetail(f"dctDay:{dctDay}")

        Logger.printDetail(f"dctQuarter:{dctQuarter}")
        Logger.printDetail(f"dctSeason:{dctSeason}")
        Logger.printDetail(f"dctWeekday:{dctWeekday}")
        Logger.printDetail(f"dctWeek:{dctWeek}")

    else:
        Logger.printDetail("No DCT available...")

    # check if value_i has month, day, season, week (otherwise no UNDEF-year is possible)
    viHasMonth = False
    viHasDay = False
    viHasSeason = False
    viHasWeek = False
    viHasQuarter = False
    viHasHalf = False
    viThisMonth = 0
    viThisDay = 0
    viThisSeason = ""
    viThisQuarter = ""
    viThisHalf = ""

    valueParts = ambigString.split("-")

    # check if UNDEF-year or UNDEF-century
    if ambigString.startswith("UNDEF-year") or ambigString.startswith("UNDEF-century"):
        offset = 2
    else:
        offset = 1

    if len(valueParts) > offset:
        # get vi month
        if re.match("\d\d", valueParts[offset]):
            viHasMonth = True
            viThisMonth = int(valueParts[offset])

        # get vi season
        elif valueParts[offset] in ["SP", "SU", "FA", "WI"]:
            viHasSeason = True
            viThisSeason = valueParts[offset]

        # get v1 quarter
        elif valueParts[offset] in ["Q1", "Q2", "Q3", "Q4"]:
            viThisQuarter = True
            viThisQuarter = valueParts[offset]

        # get v1 quarter
        elif valueParts[offset] in ["H1", "H2"]:
            viHasHalf = True
            viThisHalf = valueParts[offset]

        # get vi day
        if len(valueParts) > offset + 1 and re.match("\d\d", valueParts[offset + 1]):
            viHasDay = True
            viThisDay = int(valueParts[offset + 1][0:2])

    Logger.printDetail(f"viMonth:{viHasMonth} ({viThisMonth})")
    Logger.printDetail(f"viDay:{viHasDay} ({viThisDay})")
    Logger.printDetail(f"viSeason:{viHasSeason} ({viThisSeason})")
    Logger.printDetail(f"viQuarter:{viHasQuarter} ({viThisQuarter})")
    Logger.printDetail(f"viHalf:{viHasHalf} ({viThisHalf})")

    # get the last tense (depending on the part of speech tags used in front or behind the expression)
    last_used_tense = ContextAnalyzer.getLastTense(context, language, spacyModel)
    Logger.printDetail(f"lastTense:{last_used_tense}")

    # ////////////////////////
    #  DISAMBIGUATION PHASE //
    # ////////////////////////

    # //////////////////////////////////////////////////
    #  IF YEAR IS COMPLETELY UNSPECIFIED (UNDEF-year) //
    # //////////////////////////////////////////////////

    valueNew = ambigString
    if ambigString.startswith("UNDEF-year"):
        Logger.printDetail(f"Resolve unknown year")
        newYearValue = str(dctYear)

        # vi has month (ignore day)
        if viHasMonth and not viHasSeason:
            # WITH DOCUMENT CREATION TIME
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                # Tense is FUTURE
                if last_used_tense in ["FUTURE", "PRESENTFUTURE"]:
                    # if dct-month is larger than vi-month, than add 1 to dct-year
                    if dctMonth > viThisMonth:
                        newYearValue = str(dctYear + 1)
                # Tense is PAST
                if last_used_tense == "PAST":
                    # if dct-month is smaller than vi month, than substrate 1 from dct-year
                    if dctMonth < viThisMonth:
                        newYearValue = str(dctYear - 1)

            else:
                newYearValue = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )

        # vi has quarter
        if viHasQuarter:
            # WITH DOCUMENT CREATION TIME
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                # Tense is FUTURE
                if last_used_tense in ["FUTURE", "PRESENTFUTURE"]:
                    if int(dctQuarter[1]) < int(viThisQuarter[1]):
                        newYearValue = str(dctYear + 1)
                # Tense is PAST
                if last_used_tense == "PAST":
                    if int(dctQuarter[1]) < int(viThisQuarter[1]):
                        newYearValue = str(dctYear - 1)
                # IF NOT TENSE IS FOUND
                if last_used_tense == "":
                    if documentTypeColloquial:
                        # IN COLLOQUIAL: future temporal expression
                        if int(dctQuarter[1]) < int(viThisQuarter[1]):
                            newYearValue = str(dctYear + 1)
                    else:
                        # IN NEWS: past temporal expression
                        if int(dctQuarter[1]) < int(viThisQuarter[1]):
                            newYearValue = str(dctYear - 1)

            # WITHOUT DOCUMENT CREATION TIME
            else:
                newYearValue = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )

        # vi has half
        if viHasHalf:
            # WITH DOCUMENT CREATION TIME
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                # Tense is FUTURE
                if last_used_tense in ["FUTURE", "PRESENTFUTURE"]:
                    if int(dctHalf[1]) < int(viThisHalf[1]):
                        newYearValue = str(dctYear + 1)
                # Tense is PAST
                if last_used_tense == "PAST":
                    if int(dctHalf[1]) < int(viThisHalf[1]):
                        newYearValue = str(dctYear - 1)
                # IF NOT TENSE IS FOUND
                if last_used_tense == "":
                    if documentTypeColloquial:
                        # IN COLLOQUIAL: future temporal expression
                        if int(dctHalf[1]) < int(viThisHalf[1]):
                            newYearValue = str(dctYear + 1)
                    else:
                        # IN NEWS: past temporal expression
                        if int(dctHalf[1]) < int(viThisHalf[1]):
                            newYearValue = str(dctYear - 1)

            # WITHOUT DOCUMENT CREATION TIME
            else:
                newYearValue = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )

        # vi has season
        if not viHasMonth and not viHasDay and not viHasSeason:
            # TODO check tenses?
            # WITH DOCUMENT CREATION TIME
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                newYearValue = dctYear
            # WITHOUT DOCUMENT CREATION TIME
            else:
                newYearValue = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )

        # vi has week
        if viHasWeek:
            # WITH DOCUMENT CREATION TIME
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                newYearValue = dctYear
            # WITHOUT DOCUMENT CREATION TIME
            else:
                newYearValue = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )

        # REPLACE THE UNDEF-YEAR WITH THE NEWLY CALCULATED YEAR AND ADD TIMEX TO INDEXES
        if newYearValue == "":
            valueNew = ambigString.replace("UNDEF-year", "XXXX", 1)
        else:
            valueNew = ambigString.replace("UNDEF-year", str(newYearValue), 1)

        Logger.printDetail(f"new value:{valueNew}")

    # /////////////////////////////////////////////////
    #  just century is unspecified (UNDEF-century86) //
    # /////////////////////////////////////////////////

    elif ambigString.startswith("UNDEF-century"):
        Logger.printDetail(f"Resolve unknown century")
        newCenturyValue = str(dctCentury)

        # NEWS and COLLOQUIAL DOCUMENTS
        if (
            (documentTypeNews or documentTypeColloquial or documentTypeScientific)
            and dctAvailable
            and ambigString != "UNDEF-century"
        ):
            viThisDecade = int(ambigString[13:14])

            newCenturyValue = str(dctCentury)

            # Tense is FUTURE
            if last_used_tense in ["FUTURE", "PRESENTFUTURE"]:
                if int(viThisDecade) < int(dctDecade):
                    newCenturyValue = str(dctCentury + 1)
                else:
                    newCenturyValue = str(dctCentury)
            # Tense is PAST
            if last_used_tense == "PAST":
                if int(viThisDecade) < int(dctDecade):
                    newCenturyValue = str(dctCentury - 1)
                else:
                    newCenturyValue = str(dctCentury)

        # NARRATIVE DOCUMENTS
        else:
            newCenturyValue = ContextAnalyzer.getLastMentionedX(
                linearDates, "century", language, normalization
            )
            if not newCenturyValue.startswith("BC"):
                if re.match("\d\d", newCenturyValue) and int(newCenturyValue[0:2]) < 10:
                    newCenturyValue = "00"
            else:
                newCenturyValue = "00"

        if newCenturyValue == "":
            if not documentTypeNarrative:
                # always assume that sixties, twenties, and so on are 19XX if not century found (LREC change)
                valueNew = ambigString.replace("UNDEF-century", "19", 1)
            # LREC change: assume in narrative-style documents that if no other century was mentioned before,
            # 1st century
            else:
                valueNew = ambigString.replace("UNDEF-century", "00", 1)

        else:
            valueNew = ambigString.replace("UNDEF-century", str(newCenturyValue), 1)

        # always assume that sixties, twenties, and so on are 19XX -- if not narrative document (LREC change)
        if re.search("\d\d\d\d", valueNew) and not documentTypeNarrative:
            valueNew = "19" + valueNew[2:]

        Logger.printDetail(f"new value:{valueNew}")

    # //////////////////////////////////////////////////
    #  CHECK IMPLICIT EXPRESSIONS STARTING WITH UNDEF //
    # //////////////////////////////////////////////////

    elif ambigString.startswith("UNDEF"):
        Logger.printDetail(f"Resolve other expression starting with UNDEF")
        valueNew = ambigString

        if re.match("^UNDEF-REFDATE$", ambigString):
            if len(linearDates) > 0:
                anyDate = linearDates[-1]
                valueNew = anyDate
            else:
                valueNew = "XXXX-XX-XX"

            # ////////////////
            #  TO CALCULATE //
            # ////////////////
            # year to calculate

        elif re.search(
            "^UNDEF-(this|REFUNIT|REF)-(.*?)-(MINUS|PLUS)-([0-9]+).*", ambigString
        ):
            m_iter = re.finditer(
                "(UNDEF-(this|REFUNIT|REF)-(.*?)-(MINUS|PLUS)-([0-9]+)).*", ambigString
            )
            for mr in m_iter:
                checkUndef = mr.group(1)
                ltn = mr.group(2)
                unit = mr.group(3)
                op = mr.group(4)
                sDiff = mr.group(5)

                diff = 0
                try:
                    diff = int(sDiff)
                except:
                    Logger.printError("Expression difficult to normalize: ")
                    Logger.printError(ambigString)
                    Logger.printError(
                        sDiff + " probably too long for parsing as integer."
                    )
                    Logger.printError("set normalized value as PAST_REF / FUTURE_REF:")
                    if op == "PLUS":
                        valueNew = "FUTURE_REF"
                    else:
                        valueNew = "PAST_REF"
                    break

                # do the processing for SCIENTIFIC documents (TPZ identification could be improved)
                if documentTypeScientific:
                    opSymbol = "+" if op == "PLUS" else "-"

                    if unit == "year":
                        diffString = str(diff).zfill(4)
                        valueNew = "TPZ" + opSymbol + diffString

                    elif unit == "month":
                        diffString = "0000-" + str(diff).zfill(2)
                        valueNew = "TPZ" + opSymbol + diffString

                    elif unit == "week":
                        diffString = "0000-W" + str(diff).zfill(2)
                        valueNew = "TPZ" + opSymbol + diffString

                    elif unit == "day":
                        diffString = "0000-00-" + str(diff).zfill(2)
                        valueNew = "TPZ" + opSymbol + diffString

                    elif unit == "hour":
                        diffString = "0000-00-00T" + str(diff).zfill(2)
                        valueNew = "TPZ" + opSymbol + diffString

                    elif unit == "minute":
                        diffString = "0000-00-00T00:" + str(diff).zfill(2)
                        valueNew = "TPZ" + opSymbol + diffString

                    elif unit == "second":
                        diffString = "0000-00-00T00:00" + str(diff).zfill(2)
                        valueNew = "TPZ" + opSymbol + diffString

                else:
                    # check for REFUNIT (only allowed for "year")

                    if ltn == "REFUNIT" and unit == "year":
                        dateWithYear = ContextAnalyzer.getLastMentionedX(
                            linearDates, "dateYear", language, normalization
                        )
                        year = dateWithYear
                        if dateWithYear == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX")
                        else:
                            if dateWithYear.startswith("BC"):
                                year = dateWithYear[0:6]
                            else:
                                year = dateWithYear[0:4]
                            if op == "MINUS":
                                diff = diff * -1

                            yearNew = DateCalculator.getXNextYear(dateWithYear, diff)
                            rest = dateWithYear.substring[4]
                            valueNew = valueNew.replace(checkUndef, str(yearNew + rest))

                    # REF and this are handled here
                    if unit == "century":
                        if (
                            (
                                documentTypeNews
                                or documentTypeColloquial
                                or documentTypeScientific
                            )
                            and dctAvailable
                            and ltn == "this"
                        ):
                            century = dctCentury
                            if op == "MINUS":
                                century = dctCentury - diff
                            elif op == "PLUS":
                                century = dctCentury + diff
                            valueNew = valueNew.replace(checkUndef, str(century))

                        else:
                            lmCentury = ContextAnalyzer.getLastMentionedX(
                                linearDates, "century", language, normalization
                            )
                            if lmCentury == "":
                                valueNew = valueNew.replace(checkUndef, "XX")
                            else:
                                if op == "MINUS":
                                    diff = (-1) * diff
                                lmCentury = DateCalculator.getXNextCentury(
                                    lmCentury, diff
                                )
                                valueNew = valueNew.replace(checkUndef, str(lmCentury))

                    elif unit == "decade":
                        if (
                            (
                                documentTypeNews
                                or documentTypeColloquial
                                or documentTypeScientific
                            )
                            and dctAvailable
                            and ltn == "this"
                        ):
                            dctDecadeLong = int(str(dctCentury) + str(dctDecade))
                            decade = dctDecadeLong
                            if op == "MINUS":
                                decade = dctDecadeLong - diff
                            elif op == "PLUS":
                                decade = dctDecadeLong + diff
                            valueNew = valueNew.replace(checkUndef, str(decade) + "X")

                        else:
                            lmDecade = ContextAnalyzer.getLastMentionedX(
                                linearDates, "decade", language, normalization
                            )
                            if lmDecade == "":
                                valueNew = valueNew.replace(checkUndef, "XXX")
                            else:
                                if op == "MINUS":
                                    diff = (-1) * diff
                                lmDecade = DateCalculator.getXNextDecade(lmDecade, diff)
                                valueNew = valueNew.replace(checkUndef, str(lmDecade))

                    elif unit == "year":
                        if (
                            (
                                documentTypeNews
                                or documentTypeColloquial
                                or documentTypeScientific
                            )
                            and dctAvailable
                            and ltn == "this"
                        ):
                            intValue = dctYear
                            if op == "MINUS":
                                intValue = dctYear - diff
                            elif op == "PLUS":
                                intValue = dctYear + diff
                            valueNew = valueNew.replace(checkUndef, str(intValue))

                        else:
                            lmYear = ContextAnalyzer.getLastMentionedX(
                                linearDates, "year", language, normalization
                            )
                            if lmYear == "":
                                valueNew = valueNew.replace(checkUndef, "XXXX")
                            else:
                                if op == "MINUS":
                                    diff = (-1) * diff
                                lmYear = DateCalculator.getXNextYear(lmYear, diff)
                                valueNew = valueNew.replace(checkUndef, str(lmYear))
                            # TODO BC years

                    elif unit == "quarter":
                        if (
                            (
                                documentTypeNews
                                or documentTypeColloquial
                                or documentTypeScientific
                            )
                            and dctAvailable
                            and ltn == "this"
                        ):
                            intYear = dctYear
                            intQuarter = int(dctQuarter[1])
                            diffQuarters = diff % 4
                            diff = diff - diffQuarters
                            diffYears = diff / 4
                            if op == "MINUS":
                                diffQuarters = diffQuarters * (-1)
                                diffYears = diffYears * (-1)
                                intYear = int(intYear + diffYears)
                                intQuarter = int(intQuarter + diffQuarters)
                            lmQUarter = str(intYear).zfill(4) + "-Q" + str(intQuarter)
                            valueNew = valueNew.replace(checkUndef, str(lmQUarter))

                        else:
                            lmQuarter = ContextAnalyzer.getLastMentionedX(
                                linearDates, "quarter", language, normalization
                            )
                            if lmQuarter == "":
                                valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                            else:
                                intYear = int(lmQuarter[0:4])
                                intQuarter = int(lmQuarter[6])
                                diffQuarters = diff % 4
                                diff = diff - diffQuarters
                                diffYears = diff / 4
                                if op == "MINUS":
                                    diffQuarters = diffQuarters * (-1)
                                    diffYears = diffYears * (-1)
                                intYear = int(intYear + diffYears)
                                intQuarter = int(intQuarter + diffQuarters)
                                lmQUarter = (
                                    str(intYear).zfill(4) + "-Q" + str(intQuarter)
                                )
                                valueNew = valueNew.replace(checkUndef, str(lmQUarter))

                    elif unit == "month":
                        if (
                            (
                                documentTypeNews
                                or documentTypeColloquial
                                or documentTypeScientific
                            )
                            and dctAvailable
                            and ltn == "this"
                        ):
                            if op == "MINUS":
                                diff = diff * (-1)
                            lmMonth = DateCalculator.getXNextMonth(dct, diff)
                            valueNew = valueNew.replace(checkUndef, str(lmMonth))

                        else:
                            lmMonth = ContextAnalyzer.getLastMentionedX(
                                linearDates, "month", language, normalization
                            )
                            if lmMonth == "":
                                valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                            else:
                                if op == "MINUS":
                                    diff = (-1) * diff
                                lmMonth = DateCalculator.getXNextMonth(lmMonth, diff)
                                valueNew = valueNew.replace(checkUndef, str(lmMonth))

                    elif unit == "week":
                        if (
                            (
                                documentTypeNews
                                or documentTypeColloquial
                                or documentTypeScientific
                            )
                            and dctAvailable
                            and ltn == "this"
                        ):
                            if op == "MINUS":
                                diff = diff * (-1)
                            elif op == "PLUS":
                                # diff = diff * 7
                                pass
                            lmWeek = DateCalculator.getXNextWeek(dct, diff)
                            valueNew = valueNew.replace(checkUndef, str(lmWeek))

                        else:
                            lmDay = ContextAnalyzer.getLastMentionedX(
                                linearDates, "day", language, normalization
                            )
                            if lmDay == "":
                                valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                            else:
                                if op == "MINUS":
                                    diff = diff * 7 * (-1)
                                elif op == "PLUS":
                                    diff = diff * 7
                                valueNew = valueNew.replace(
                                    checkUndef,
                                    str(DateCalculator.getXNextDay(lmDay, diff)),
                                )

                    elif unit == "day":
                        if (
                            (
                                documentTypeNews
                                or documentTypeColloquial
                                or documentTypeScientific
                            )
                            and dctAvailable
                            and ltn == "this"
                        ):
                            if op == "MINUS":
                                diff = diff * (-1)
                            lmDay = DateCalculator.getXNextDay(dct, diff)
                            valueNew = valueNew.replace(checkUndef, str(lmDay))

                        else:
                            lmDay = ContextAnalyzer.getLastMentionedX(
                                linearDates, "day", language, normalization
                            )
                            if lmDay == "":
                                valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                            else:
                                if op == "MINUS":
                                    diff = (-1) * diff
                                lmDay = DateCalculator.getXNextDay(lmDay, diff)
                                valueNew = valueNew.replace(checkUndef, str(lmDay))

        # century
        elif ambigString.startswith("UNDEF-last-century"):
            checkUndef = "UNDEF-last-century"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctCentury - 1).zfill(2))
            else:
                lmCentury = ContextAnalyzer.getLastMentionedX(
                    linearDates, "century", language, normalization
                )
                if lmCentury == "":
                    valueNew = valueNew.replace(checkUndef, "XX")
                else:
                    lmCentury = DateCalculator.getXNextCentury(lmCentury, -1)
                    valueNew = valueNew.replace(checkUndef, str(lmCentury))

        elif ambigString.startswith("UNDEF-this-century"):
            checkUndef = "UNDEF-this-century"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctCentury).zfill(2))
            else:
                lmCentury = ContextAnalyzer.getLastMentionedX(
                    linearDates, "century", language, normalization
                )
                if lmCentury == "":
                    valueNew = valueNew.replace(checkUndef, "XX")
                else:
                    valueNew = valueNew.replace(checkUndef, str(lmCentury))

        elif ambigString.startswith("UNDEF-next-century"):
            checkUndef = "UNDEF-next-century"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctCentury + 1).zfill(2))
            else:
                lmCentury = ContextAnalyzer.getLastMentionedX(
                    linearDates, "century", language, normalization
                )
                if lmCentury == "":
                    valueNew = valueNew.replace(checkUndef, "XX")
                else:
                    lmCentury = DateCalculator.getXNextCentury(lmCentury, +1)
                    valueNew = valueNew.replace(checkUndef, str(lmCentury))

        # decade
        elif ambigString.startswith("UNDEF-last-decade"):
            checkUndef = "UNDEF-last-decade"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctYear - 10).zfill(4)[0:3])
            else:
                lmDecade = ContextAnalyzer.getLastMentionedX(
                    linearDates, "decade", language, normalization
                )
                if lmDecade == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX")
                else:
                    lmDecade = DateCalculator.getXNextDecade(lmDecade, -1)
                    valueNew = valueNew.replace(checkUndef, str(lmDecade))

        elif ambigString.startswith("UNDEF-this-decade"):
            checkUndef = "UNDEF-this-decade"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctYear).zfill(4)[0:3])
            else:
                lmDecade = ContextAnalyzer.getLastMentionedX(
                    linearDates, "decade", language, normalization
                )
                if lmDecade == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX")
                else:
                    valueNew = valueNew.replace(checkUndef, str(lmDecade))

        elif ambigString.startswith("UNDEF-next-decade"):
            checkUndef = "UNDEF-next-decade"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctYear + 10).zfill(4)[0:3])
            else:
                lmDecade = ContextAnalyzer.getLastMentionedX(
                    linearDates, "decade", language, normalization
                )
                if lmDecade == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX")
                else:
                    lmDecade = DateCalculator.getXNextDecade(lmDecade, +1)
                    valueNew = valueNew.replace(checkUndef, str(lmDecade))

        # year
        elif ambigString.startswith("UNDEF-last-year"):
            checkUndef = "UNDEF-last-year"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctYear - 1).zfill(4))
            else:
                lmYear = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )
                if lmYear == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX")
                else:
                    lmYear = DateCalculator.getXNextYear(lmYear, -1)
                    valueNew = valueNew.replace(checkUndef, str(lmYear))
                if valueNew.endswith("-FY"):
                    valueNew = "FY" + valueNew[0 : min(len(valueNew), 4)]

        elif ambigString.startswith("UNDEF-this-year"):
            checkUndef = "UNDEF-this-year"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctYear))
            else:
                lmYear = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )
                if lmYear == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX")
                else:
                    valueNew = valueNew.replace(checkUndef, str(lmYear))
                if valueNew.endswith("-FY"):
                    valueNew = "FY" + valueNew[0 : min(len(valueNew), 4)]

        elif ambigString.startswith("UNDEF-next-year"):
            checkUndef = "UNDEF-next-year"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                valueNew = valueNew.replace(checkUndef, str(dctYear + 1).zfill(4))
            else:
                lmYear = ContextAnalyzer.getLastMentionedX(
                    linearDates, "year", language, normalization
                )
                if lmYear == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX")
                else:
                    lmYear = DateCalculator.getXNextYear(lmYear, +1)
                    valueNew = valueNew.replace(checkUndef, str(lmYear))
                if valueNew.endswith("-FY"):
                    valueNew = "FY" + valueNew[0 : min(len(valueNew), 4)]

        # month
        elif ambigString.startswith("UNDEF-last-month"):
            checkUndef = "UNDEF-last-month"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmMonth = DateCalculator.getXNextMonth(dct, -1)
                valueNew = valueNew.replace(checkUndef, lmMonth)
            else:
                lmMonth = ContextAnalyzer.getLastMentionedX(
                    linearDates, "month", language, normalization
                )
                if lmMonth == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                else:
                    lmMonth = DateCalculator.getXNextMonth(lmMonth, -1)
                    valueNew = valueNew.replace(checkUndef, str(lmMonth))

        elif ambigString.startswith("UNDEF-this-month"):
            checkUndef = "UNDEF-this-month"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmMonth = f"{str(dctYear).zfill(4)}-{str(dctMonth).zfill(2)}"
                valueNew = valueNew.replace(checkUndef, lmMonth)
            else:
                lmMonth = ContextAnalyzer.getLastMentionedX(
                    linearDates, "month", language, normalization
                )
                if lmMonth == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                else:
                    valueNew = valueNew.replace(checkUndef, str(lmMonth))

        elif ambigString.startswith("UNDEF-next-month"):
            checkUndef = "UNDEF-next-month"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmMonth = DateCalculator.getXNextMonth(dct, +1)
                valueNew = valueNew.replace(checkUndef, lmMonth)
            else:
                lmMonth = ContextAnalyzer.getLastMentionedX(
                    linearDates, "month", language, normalization
                )
                if lmMonth == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                else:
                    lmMonth = DateCalculator.getXNextMonth(lmMonth, +1)
                    valueNew = valueNew.replace(checkUndef, str(lmMonth))

        # day
        elif ambigString.startswith("UNDEF-last-day"):
            checkUndef = "UNDEF-last-day"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmDay = DateCalculator.getXNextDay(dct, -1)
                valueNew = valueNew.replace(checkUndef, lmDay)
            else:
                lmDay = ContextAnalyzer.getLastMentionedX(
                    linearDates, "day", language, normalization
                )
                if lmDay == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                else:
                    lmDay = DateCalculator.getXNextDay(lmDay, -1)
                    valueNew = valueNew.replace(checkUndef, str(lmDay))

        elif ambigString.startswith("UNDEF-this-day"):
            checkUndef = "UNDEF-this-day"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmDay = f"{str(dctYear).zfill(4)}-{str(dctMonth).zfill(2)}-{str(dctDay).zfill(2)}"
                valueNew = valueNew.replace(checkUndef, lmDay)
            else:
                lmDay = ContextAnalyzer.getLastMentionedX(
                    linearDates, "day", language, normalization
                )
                if lmDay == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                else:
                    valueNew = valueNew.replace(checkUndef, str(lmDay))
                if ambigString == "UNDEF-this-day":
                    valueNew = "PRESENT_REF"

        elif ambigString.startswith("UNDEF-next-day"):
            checkUndef = "UNDEF-next-day"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmDay = DateCalculator.getXNextDay(dct, +1)
                valueNew = valueNew.replace(checkUndef, lmDay)
            else:
                lmDay = ContextAnalyzer.getLastMentionedX(
                    linearDates, "day", language, normalization
                )
                if lmDay == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                else:
                    lmDay = DateCalculator.getXNextDay(lmDay, +1)
                    valueNew = valueNew.replace(checkUndef, str(lmDay))

        # week
        elif ambigString.startswith("UNDEF-last-week"):
            checkUndef = "UNDEF-last-week"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmWeek = DateCalculator.getXNextWeek(dct, -1)
                valueNew = valueNew.replace(checkUndef, lmWeek)
            else:
                lmWeek = ContextAnalyzer.getLastMentionedX(
                    linearDates, "week", language, normalization
                )
                if lmWeek == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-WXX")
                else:
                    lmWeek = DateCalculator.getXNextWeek(lmWeek, -1)
                    valueNew = valueNew.replace(checkUndef, str(lmWeek))

        elif ambigString.startswith("UNDEF-this-week"):
            checkUndef = "UNDEF-this-week"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmWeek = f"{str(dctYear).zfill(4)}-W{str(dctWeek).zfill(2)}"
                valueNew = valueNew.replace(checkUndef, lmWeek)
            else:
                lmWeek = ContextAnalyzer.getLastMentionedX(
                    linearDates, "week", language, normalization
                )
                if lmWeek == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-WXX")
                else:
                    valueNew = valueNew.replace(checkUndef, str(lmWeek))
        elif ambigString.startswith("UNDEF-next-week"):
            checkUndef = "UNDEF-next-week"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmWeek = DateCalculator.getXNextWeek(dct, +1)
                valueNew = valueNew.replace(checkUndef, lmWeek)
            else:
                lmWeek = ContextAnalyzer.getLastMentionedX(
                    linearDates, "week", language, normalization
                )
                if lmWeek == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-WXX")
                else:
                    lmWeek = DateCalculator.getXNextWeek(lmWeek, +1)
                    valueNew = valueNew.replace(checkUndef, str(lmWeek))

        # quarter
        elif ambigString.startswith("UNDEF-last-quarter"):
            checkUndef = "UNDEF-last-quarter"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                if dctQuarter == "Q1":
                    lmQuarter = f"{str(dctYear-1).zfill(4)}-Q4"
                    valueNew = valueNew.replace(checkUndef, str(lmQuarter))
                else:
                    newQuarter = int(dctQuarter[2]) - 1
                    lmQuarter = f"{str(dctYear).zfill(4)}-Q{newQuarter}"
                    valueNew = valueNew.replace(checkUndef, str(lmQuarter))
            else:
                lmQuarter = ContextAnalyzer.getLastMentionedX(
                    linearDates, "quarter", language, normalization
                )
                if lmQuarter == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-QX")
                else:
                    lmQuarterOnly = int(lmQuarter[6:7])
                    lmYearOnly = int(lmQuarter[0:4])
                    if lmQuarterOnly == 1:
                        lmQuarter = f"{str(lmYearOnly-1).zfill(4)}-Q4"
                        valueNew = valueNew.replace(checkUndef, str(lmQuarter))
                    else:
                        lmQuarter = f"{str(lmYearOnly).zfill(4)}-Q{lmQuarterOnly-1}"
                        valueNew = valueNew.replace(checkUndef, str(lmQuarter))

        elif ambigString.startswith("UNDEF-this-quarter"):
            checkUndef = "UNDEF-this-quarter"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                lmQuarter = f"{str(dctYear).zfill(4)}-Q{dctQuarter}"
                valueNew = valueNew.replace(checkUndef, lmQuarter)
            else:
                lmQuarter = ContextAnalyzer.getLastMentionedX(
                    linearDates, "quarter", language, normalization
                )
                if lmQuarter == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-QX")
                else:
                    valueNew = valueNew.replace(checkUndef, str(lmQuarter))

        elif ambigString.startswith("UNDEF-last-quarter"):
            checkUndef = "UNDEF-last-quarter"
            if (
                documentTypeNews or documentTypeColloquial or documentTypeScientific
            ) and dctAvailable:
                if dctQuarter == "Q4":
                    lmQuarter = f"{str(dctYear+1).zfill(4)}-Q1"
                    valueNew = valueNew.replace(checkUndef, str(lmQuarter))
                else:
                    newQuarter = int(dctQuarter[2]) + 1
                    lmQuarter = f"{str(dctYear).zfill(4)}-Q{newQuarter}"
                    valueNew = valueNew.replace(checkUndef, str(lmQuarter))
            else:
                lmQuarter = ContextAnalyzer.getLastMentionedX(
                    linearDates, "quarter", language, normalization
                )
                if lmQuarter == "":
                    valueNew = valueNew.replace(checkUndef, "XXXX-QX")
                else:
                    lmQuarterOnly = int(lmQuarter[6:7])
                    lmYearOnly = int(lmQuarter[0:4])
                    if lmQuarterOnly == 4:
                        lmQuarter = f"{str(lmYearOnly+1).zfill(4)}-Q1"
                        valueNew = valueNew.replace(checkUndef, str(lmQuarter))
                    else:
                        lmQuarter = f"{str(lmYearOnly).zfill(4)}-Q{lmQuarterOnly+1}"
                        valueNew = valueNew.replace(checkUndef, str(lmQuarter))

        # Month names
        elif re.search(
            "^UNDEF-(last|this|next)-(january|february|march|april|may|june|july|august|september|october|november|december)(.*)",
            ambigString,
        ):
            m_iter = re.finditer(
                "^UNDEF-(last|this|next)-(january|february|march|april|may|june|july|august|september|october|november|december)(.*)",
                ambigString,
            )
            for mr in m_iter:
                rest = mr.group(3)
                day = 0

                m_rest_iter = re.finditer("-(\d\d)", rest)
                for mr_rest in m_rest_iter:
                    day = int(mr_rest.group(1))

                checkUndef = mr.group(0)
                ltn = mr.group(1)
                newMonth = normalization["reMonth"][mr.group(2)]
                intNewMonth = int(newMonth)

                if ltn == "last":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        # check day if dct-month and newMonth are equal
                        if dctMonth == intNewMonth and day != 0:
                            if dctDay > day:
                                lmMonth = (
                                    str(dctYear).zfill(4) + "-" + str(newMonth).zfill(2)
                                )
                                valueNew = valueNew.replace(checkUndef, lmMonth)
                            else:
                                lmMonth = (
                                    str(dctYear - 1).zfill(4)
                                    + "-"
                                    + str(newMonth).zfill(2)
                                )
                                valueNew = valueNew.replace(checkUndef, lmMonth)
                        elif dctMonth <= intNewMonth:
                            lmMonth = (
                                str(dctYear - 1).zfill(4) + "-" + str(newMonth).zfill(2)
                            )
                            valueNew = valueNew.replace(checkUndef, lmMonth)
                        else:
                            lmMonth = (
                                str(dctYear).zfill(4) + "-" + str(newMonth).zfill(2)
                            )
                            valueNew = valueNew.replace(checkUndef, lmMonth)

                    else:
                        lmMonth = ContextAnalyzer.getLastMentionedX(
                            linearDates, "month-with-details", language, normalization
                        )
                        if lmMonth == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                        else:
                            lmMonthInt = int(lmMonth[5:7])
                            lmDayInt = 0
                            if len(lmMonth) > 9 and re.match("\\d\\d", lmMonth[8:10]):
                                lmDayInt = int(lmMonth[8:10])

                            if lmMonthInt == intNewMonth and lmDayInt != 0 and day != 0:
                                if lmDayInt > day:
                                    lmMonthT = lmMonth[0:4] + "-" + newMonth
                                    valueNew = valueNew.replace(checkUndef, lmMonthT)
                                else:
                                    lmMonthT = (
                                        str(int(lmMonth[0:4]) - 1).zfill(4)
                                        + "-"
                                        + newMonth
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmMonthT)

                            if lmMonthInt <= intNewMonth:
                                lmMonthT = (
                                    str(int(lmMonth[0:4]) - 1).zfill(4) + "-" + newMonth
                                )
                                valueNew = valueNew.replace(checkUndef, lmMonthT)
                            else:
                                lmMonthT = lmMonth[0:4] + "-" + newMonth
                                valueNew = valueNew.replace(checkUndef, lmMonthT)

                elif ltn == "this":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        lmMonth = str(dctYear).zfill(4) + "-" + newMonth
                        valueNew = valueNew.replace(checkUndef, lmMonth)
                    else:
                        lmMonth = ContextAnalyzer.getLastMentionedX(
                            linearDates, "month-with-details", language, normalization
                        )
                        if lmMonth == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                        else:
                            lmMonth = str(dctYear).zfill(4) + "-" + newMonth
                            valueNew = valueNew.replace(checkUndef, lmMonth)

                elif ltn == "next":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        # check day if dct-month and newMonth are equal
                        if dctMonth == intNewMonth and day != 0:
                            if dctDay < day:
                                lmMonth = (
                                    str(dctYear).zfill(4) + "-" + str(newMonth).zfill(2)
                                )
                                valueNew = valueNew.replace(checkUndef, lmMonth)
                            else:
                                lmMonth = (
                                    str(dctYear + 1).zfill(4)
                                    + "-"
                                    + str(newMonth).zfill(2)
                                )
                                valueNew = valueNew.replace(checkUndef, lmMonth)
                        elif dctMonth >= intNewMonth:
                            lmMonth = (
                                str(dctYear + 1).zfill(4) + "-" + str(newMonth).zfill(2)
                            )
                            valueNew = valueNew.replace(checkUndef, lmMonth)
                        else:
                            lmMonth = (
                                str(dctYear).zfill(4) + "-" + str(newMonth).zfill(2)
                            )
                            valueNew = valueNew.replace(checkUndef, lmMonth)

                    else:
                        lmMonth = ContextAnalyzer.getLastMentionedX(
                            linearDates, "month-with-details", language, normalization
                        )
                        if lmMonth == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                        else:
                            lmMonthInt = int(lmMonth[5:7])

                            if lmMonthInt >= intNewMonth:
                                lmMonthT = (
                                    str(int(lmMonth[0:4]) + 1).zfill(4) + "-" + newMonth
                                )
                                valueNew = valueNew.replace(checkUndef, lmMonthT)
                            else:
                                lmMonthT = lmMonth[0:4] + "-" + newMonth
                                valueNew = valueNew.replace(checkUndef, lmMonthT)

        # SEASONS NAMES
        elif re.search("^UNDEF-(last|this|next)-(SP|SU|FA|WI).*", ambigString):
            m_iter = re.finditer(
                "(UNDEF-(last|this|next)-(SP|SU|FA|WI)).*", ambigString
            )
            for mr in m_iter:
                checkUndef = mr.group(1)
                ltn = mr.group(2)
                newSeason = mr.group(3)

                if ltn == "last":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        if dctSeason == "SP":
                            lmSeason = str(dctYear - 1).zfill(4) + "-" + newSeason
                            valueNew = valueNew.replace(checkUndef, lmSeason)
                        elif dctSeason == "SU":
                            if newSeason == "SP":
                                lmSeason = str(dctYear).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            else:
                                lmSeason = str(dctYear - 1).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                        elif dctSeason == "FA":
                            if newSeason in ["SP", "SU"]:
                                lmSeason = str(dctYear).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            else:
                                lmSeason = str(dctYear - 1).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                        elif dctSeason == "WI":
                            if newSeason == "WI":
                                lmSeason = str(dctYear - 1).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            elif dctMonth < 12:
                                lmSeason = str(dctYear - 1).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            else:
                                lmSeason = str(dctYear).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)

                    # NARRATIVE DOCUMENT
                    else:
                        lmSeason = ContextAnalyzer.getLastMentionedX(
                            linearDates, "season", language, normalization
                        )
                        if lmSeason == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                        else:
                            if lmSeason[5:7] == "SP":
                                lmSeason = (
                                    str(int(lmSeason[0:4]) - 1).zfill(4)
                                    + "-"
                                    + newSeason
                                )
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            elif lmSeason[5:7] == "SU":
                                if lmSeason[5:7] == "SP":
                                    lmSeason = (
                                        str(int(lmSeason[0:4])).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                                else:
                                    lmSeason = (
                                        str(int(lmSeason[0:4]) - 1).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                            elif lmSeason[5:7] == "FA":
                                if lmSeason[5:7] in ["SP", "SU"]:
                                    lmSeason = (
                                        str(int(lmSeason[0:4])).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                                else:
                                    lmSeason = (
                                        str(int(lmSeason[0:4]) - 1).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                            elif lmSeason[5:7] == "WI":
                                if lmSeason[5:7] == "WI":
                                    lmSeason = (
                                        str(int(lmSeason[0:4]) - 1).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                                else:
                                    lmSeason = (
                                        str(int(lmSeason[0:4])).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)

                elif ltn == "this":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        # TODO include tense of sentence?
                        lmSeason = str(dctYear).zfill(4) + "-" + newSeason
                        valueNew = valueNew.replace(checkUndef, lmSeason)
                    else:
                        # TODO include tense of sentence?
                        lmSeason = ContextAnalyzer.getLastMentionedX(
                            linearDates, "season", language, normalization
                        )
                        if lmSeason == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                        else:
                            lmSeason = str(lmSeason[0:4]).zfill(4) + "-" + newSeason
                            valueNew = valueNew.replace(checkUndef, lmSeason)

                if ltn == "lanextst":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        if dctSeason == "SP":
                            if newSeason == "SP":
                                lmSeason = str(dctYear + 1).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            else:
                                lmSeason = str(dctYear).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                        elif dctSeason == "SU":
                            if newSeason == ["SP", "SU"]:
                                lmSeason = str(dctYear + 1).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            else:
                                lmSeason = str(dctYear).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                        elif dctSeason == "FA":
                            if newSeason in ["SP", "SU", "FA"]:
                                lmSeason = str(dctYear + 1).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                            else:
                                lmSeason = str(dctYear).zfill(4) + "-" + newSeason
                                valueNew = valueNew.replace(checkUndef, lmSeason)
                        elif dctSeason == "WI":
                            lmSeason = str(dctYear + 1).zfill(4) + "-" + newSeason
                            valueNew = valueNew.replace(checkUndef, lmSeason)

                    # NARRATIVE DOCUMENT
                    else:
                        lmSeason = ContextAnalyzer.getLastMentionedX(
                            linearDates, "season", language, normalization
                        )
                        if lmSeason == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX")
                        else:
                            if lmSeason[5:7] == "SP":
                                if newSeason == "SP":
                                    lmSeason = (
                                        str(int(lmSeason[0:4]) + 1).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                                else:
                                    lmSeason = (
                                        str(int(lmSeason[0:4])).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                            elif lmSeason[5:7] == "SU":
                                if lmSeason[5:7] in ["SP", "SU"]:
                                    lmSeason = (
                                        str(int(lmSeason[0:4]) + 1).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                                else:
                                    lmSeason = (
                                        str(int(lmSeason[0:4])).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                            elif lmSeason[5:7] == "FA":
                                if lmSeason[5:7] in ["SP", "SU", "Fa"]:
                                    lmSeason = (
                                        str(int(lmSeason[0:4]) + 1).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                                else:
                                    lmSeason = (
                                        str(int(lmSeason[0:4])).zfill(4)
                                        + "-"
                                        + newSeason
                                    )
                                    valueNew = valueNew.replace(checkUndef, lmSeason)
                            elif lmSeason[5:7] == "WI":
                                lmSeason = (
                                    str(int(lmSeason[0:4]) + 1).zfill(4)
                                    + "-"
                                    + newSeason
                                )
                                valueNew = valueNew.replace(checkUndef, lmSeason)

        # WEEKDAY NAMES
        # TODO the calculation is strange, but works
        # TODO tense should be included?!
        elif re.search(
            "^UNDEF-(last|this|next|day)-(monday|tuesday|wednesday|thursday|friday|saturday|sunday).*",
            ambigString,
        ):
            m_iter = re.finditer(
                "(UNDEF-(last|this|next|day)-(monday|tuesday|wednesday|thursday|friday|saturday|sunday)).*",
                ambigString,
            )
            for mr in m_iter:
                checkUndef = mr.group(1)
                ltnd = mr.group(2)
                newWeekday = mr.group(3)
                newWeekdayInt = normalization["reWeekdayToInt"][newWeekday]

                if ltnd == "last":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        diff = (-1) * (dctWeekday - newWeekdayInt)
                        if diff >= 0:
                            diff = diff - 7
                        lmDay = (
                            str(dctYear).zfill(4)
                            + "-"
                            + str(dctMonth).zfill(2)
                            + "-"
                            + str(dctDay).zfill(2)
                        )
                        lmDay = DateCalculator.getXNextDay(lmDay, diff)
                        valueNew = valueNew.replace(checkUndef, str(lmDay))
                    else:
                        lmDay = ContextAnalyzer.getLastMentionedX(
                            linearDates, "day", language, normalization
                        )
                        if lmDay == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                        else:
                            lmWeekdayInt = DateCalculator.getWeekdayOfDate(lmDay)
                            diff = (-1) * (dctWeekday - lmWeekdayInt)
                            if diff >= 0:
                                diff = diff - 7
                            lmDay = DateCalculator.getXNextDay(lmDay, diff)
                            valueNew = valueNew.replace(checkUndef, str(lmDay))

                elif ltnd == "this":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        # TODO tense should be included ?!
                        diff = (-1) * (dctWeekday - newWeekdayInt)
                        if diff >= 0:
                            diff = diff - 7
                        if diff == -7:
                            diff = 0
                        lmDay = (
                            str(dctYear).zfill(4)
                            + "-"
                            + str(dctMonth).zfill(2)
                            + "-"
                            + str(dctDay).zfill(2)
                        )
                        lmDay = DateCalculator.getXNextDay(lmDay, diff)
                        valueNew = valueNew.replace(checkUndef, str(lmDay))
                    else:
                        lmDay = ContextAnalyzer.getLastMentionedX(
                            linearDates, "day", language, normalization
                        )
                        if lmDay == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                        else:
                            lmWeekdayInt = DateCalculator.getWeekdayOfDate(lmDay)
                            diff = (-1) * (dctWeekday - lmWeekdayInt)
                            if diff >= 0:
                                diff = diff - 7
                            if diff == -7:
                                diff = 0
                            lmDay = DateCalculator.getXNextDay(lmDay, diff)
                            valueNew = valueNew.replace(checkUndef, str(lmDay))

                elif ltnd == "next":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        diff = (-1) * (newWeekdayInt - dctWeekday)
                        if diff <= 0:
                            diff = diff - 7
                        lmDay = (
                            str(dctYear).zfill(4)
                            + "-"
                            + str(dctMonth).zfill(2)
                            + "-"
                            + str(dctDay).zfill(2)
                        )
                        lmDay = DateCalculator.getXNextDay(lmDay, diff)
                        valueNew = valueNew.replace(checkUndef, lmDay)
                    else:
                        lmDay = ContextAnalyzer.getLastMentionedX(
                            linearDates, "day", language, normalization
                        )
                        if lmDay == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                        else:
                            lmWeekdayInt = DateCalculator.getWeekdayOfDate(lmDay)
                            diff = (-1) * (newWeekdayInt - dctWeekday)
                            if diff <= 0:
                                diff = diff - 7
                            lmDay = DateCalculator.getXNextDay(lmDay, diff)
                            valueNew = valueNew.replace(checkUndef, lmDay)

                elif ltnd == "day":
                    if (
                        documentTypeNews
                        or documentTypeColloquial
                        or documentTypeScientific
                    ) and dctAvailable:
                        # TODO tense should be included ?!
                        diff = (-1) * (dctWeekday - newWeekdayInt)
                        if diff >= 0:
                            diff = diff - 7
                        if diff == -7:
                            diff = 0
                        if last_used_tense == "FUTURE" and diff != 0:
                            diff = diff + 7
                        if last_used_tense == "PAST":
                            pass
                        lmDay = (
                            str(dctYear).zfill(4)
                            + "-"
                            + str(dctMonth).zfill(2)
                            + "-"
                            + str(dctDay).zfill(2)
                        )
                        lmDay = DateCalculator.getXNextDay(lmDay, diff)
                        valueNew = valueNew.replace(checkUndef, lmDay)
                    else:
                        # TODO tense should be included ?!
                        lmDay = ContextAnalyzer.getLastMentionedX(
                            linearDates, "day", language, normalization
                        )
                        if lmDay == "":
                            valueNew = valueNew.replace(checkUndef, "XXXX-XX-XX")
                        else:
                            lmWeekdayInt = DateCalculator.getWeekdayOfDate(lmDay)
                            diff = (-1) * (dctWeekday - lmWeekdayInt)
                            if diff >= 0:
                                diff = diff - 7
                            if diff == -7:
                                diff = 0
                            lmDay = DateCalculator.getXNextDay(lmDay, diff)
                            valueNew = valueNew.replace(checkUndef, lmDay)

    else:
        Logger.printDetail(
            "ATTENTION: UNDEF value for: "
            + valueNew
            + " is not handled in disambiguation phase!"
        )
    return valueNew


def correctDurationValue(value):
    """Durations of a finer granularity are mapped to a coarser one if possible, e.g., "PT24H" -> "P1D".
    One may add several further corrections.
    @param value
    @return
    """

    if re.match("PT[0-9]+H", value):
        m_iter = re.finditer("PT([0-9]+)H", value)
        for mr in m_iter:
            hours = int(mr.group(1))
            if hours % 24 == 0:
                days = hours / 24
                value = "P" + str(int(days)) + "D"

    elif re.match("PT[0-9]+M", value):
        m_iter = re.finditer("PT([0-9]+)M", value)
        for mr in m_iter:
            minutes = int(mr.group(1))
            if minutes % 60 == 0:
                days = minutes / 60
                value = "PT" + str(int(days)) + "H"

    elif re.match("P[0-9]+M", value):
        m_iter = re.finditer("P([0-9]+)M", value)
        for mr in m_iter:
            months = int(mr.group(1))
            if months % 12 == 0:
                years = months / 12
                value = "P" + str(int(years)) + "Y"

    return value
