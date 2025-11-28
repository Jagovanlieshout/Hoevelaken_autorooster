import pandas as pd
from datetime import timedelta

def validate_auto_rooster(data, result):
    """
    Validates the shift assignments against all constraints.
    Returns a list of error messages if any constraints are violated.
    """
    
    print("=== VALIDATING SCHEDULE ===")
    
    errors = []

    shifts = data['shifts']
    workers = data['workers']
    onb = data['onb']

    assignments_df = result['assignments_df']
        
    
    # Drop unassigned shifts
    assignments_df = assignments_df.dropna(subset=['employee_id'])
    
    # Helper: quick lookup
    shifts_lookup = shifts.set_index('shift_id')

    # COA 7.1 exemptions: employees whose max consecutive nights differs from default 5
    COA_7_1_exempt = set()

    for _, w in workers.iterrows():
        mid = w['medewerker_id']
        max_consec = 5
        patroon = w.get('patroon') if isinstance(w, dict) else w.get('patroon') if hasattr(w, 'get') else w['patroon'] if 'patroon' in w.index else None
        voorkeur_nacht = workers.loc[workers['medewerker_id'] == mid, 'voorkeur_nacht'].iloc[0]
        
        if voorkeur_nacht == 'uitsluitend' and patroon is not None and not (isinstance(patroon, float) and pd.isna(patroon)):
            try:
                first = str(patroon).split(',')[0].strip()
                if first:
                    max_consec = int(first)
            except Exception:
                pass

        if max_consec != 5:
            COA_7_1_exempt.add(mid)
    CAO_7_4_exempt = workers[workers['voorkeur_nacht'] != 'niet']['medewerker_id'].astype(str).tolist()

    ## 1) Coverage: each shift has <= 1 assigned employee
    for sid, group in assignments_df.groupby("shift_id"):
        if len(group) > 1:
            errors.append(f"Shift {sid} assigned to multiple employees: {group['employee_id'].tolist()}")

    ## 2) At most one shift per employee per day
    for (emp, date), group in assignments_df.groupby(["employee_id", "shift_date"]):
        if len(group) > 1:
            errors.append(f"Employee {emp} works {len(group)} shifts on {date}: {group['shift_id'].tolist()}")

    ## 3) After night shift, no day/evening shift next day
    for emp, group in assignments_df.groupby("employee_id"):
        nights = group[group["is_night"]]
        for _, ns in nights.iterrows():
            next_day = ns['shift_date'] + timedelta(days=1)
            next_day_shifts = group[group['shift_date'] == next_day]
            next_day_shifts = next_day_shifts[next_day_shifts['is_night'] == False]
            if len(next_day_shifts) > 0:
                errors.append(f"Employee {emp} has day/evening shift(s) on {next_day} after a night shift on {ns['shift_date']}.")

    ## 4) Max work days per week
    for emp, group in assignments_df.groupby("employee_id"):
        max_days = int(workers.loc[workers['medewerker_id'] == emp, 'max_days_per_week'].iloc[0])
        for week, week_group in group.groupby("week"):
            worked_days = week_group['shift_date'].nunique()
            if worked_days > max_days:
                errors.append(f"Employee {emp} exceeds max work days in week {week}: {worked_days} > {max_days}")

    ## 5) Contract hours on average across all weeks
    num_weeks = assignments_df['week'].nunique()
    for emp, group in assignments_df.groupby("employee_id"):
        cap_minutes = int(workers.loc[workers['medewerker_id'] == emp, 'contract_minutes'].iloc[0])
        total_minutes = group['duration_min'].sum()
        allowed_total = cap_minutes * num_weeks
        if total_minutes > allowed_total:
            errors.append(
                f"Employee {emp} exceeds average contract hours: "
                f"{total_minutes} min worked > {allowed_total} min allowed over {num_weeks} weeks"
            )

    ## 6) No shifts overlapping with unavailable times
    for _, r in onb.iterrows():
        emp = r['Medewerker id']
        if emp not in workers['medewerker_id'].values:
            continue

        besch = r['Beschikbaarheid'].lower()
        if besch == 'beschikbaar':
            continue

        date_unb = r['Datum']
        start_unb = r.get('Beschikbaarheid_tijd_vanaf')
        end_unb = r.get('Beschikbaarheid_tijd_tm')

        emp_shifts = assignments_df[assignments_df["employee_id"] == emp]
        for _, sr in emp_shifts.iterrows():
            if sr['shift_date'] != date_unb:
                continue

            if pd.isna(start_unb) or pd.isna(end_unb):
                errors.append(f"Employee {emp} scheduled on fully unavailable day {date_unb}, shift {sr['shift_id']}.")
                continue

            overlap = not (sr['end_time'] <= start_unb or sr['start_time'] >= end_unb)
            if overlap:
                errors.append(
                    f"Employee {emp} scheduled during unavailable time on {date_unb}, "
                    f"shift {sr['shift_id']} ({sr['start_time']}–{sr['end_time']} overlaps with {start_unb}–{end_unb})."
                )

    ## 7.1) Max 5 consecutive nights
    for emp, group in assignments_df.groupby("employee_id"):
        emp_nights = group[group['is_night']].sort_values("shift_date")
        dates = emp_nights['shift_date'].tolist()
        max_consec = 7 if emp in COA_7_1_exempt else 5
        for i in range(len(dates) - max_consec):
            window = dates[i:i+max_consec+1]
            if (window[-1] - window[0]).days == max_consec:
                errors.append(f"Employee {emp} works >{max_consec} consecutive nights: {window}")

    ## 7.2) After ≥3 consecutive nights → 46h rest (next 2 days off)
    for emp, group in assignments_df.groupby("employee_id"):
        emp_nights = group[group['is_night']].sort_values("shift_date")
        night_dates = emp_nights["shift_date"].tolist()

        for i in range(len(night_dates) - 2):
            d0, d1, d2 = night_dates[i:i+3]
            if d1 == d0 + timedelta(days=1) and d2 == d1 + timedelta(days=1):
                next_day = d2 + timedelta(days=1)
                if next_day in night_dates:  # still part of a longer block
                    continue
                blocked_days = [next_day, next_day + timedelta(days=1)]
                next_shifts = group[group["shift_date"].isin(blocked_days)]
                if len(next_shifts) > 0:
                    errors.append(
                        f"Employee {emp} has shifts {next_shifts['shift_id'].tolist()} within 46h rest after nights {d0}, {d1}, {d2}."
                    )

    ## 7.3) Max 35 nights per 13 weeks
    for emp, group in assignments_df.groupby("employee_id"):
        emp_nights = group[group['is_night']]
        for start_week in range(shifts['week'].min(), shifts['week'].max()-12):
            window = emp_nights[(emp_nights['week'] >= start_week) & (emp_nights['week'] < start_week+13)]
            if len(window) > 35:
                errors.append(f"Employee {emp} has {len(window)} nights in weeks {start_week}-{start_week+12} (> 35).")

    ## 7.4) Age > 55: no night shifts
    for emp, group in assignments_df.groupby("employee_id"):
        if emp in CAO_7_4_exempt:
            continue
        leeftijd = int(workers.loc[workers['medewerker_id'] == emp, 'leeftijd'].iloc[0])
        if leeftijd > 55 and group['is_night'].any():
            errors.append(f"Employee {emp} (age {leeftijd}) assigned to night shifts: {group[group['is_night']==True]['shift_id'].tolist()}")

    if errors:
        print(f"=== VALIDATION FAILED: {len(errors)} issue(s) ===")
        for e in errors:
            print("❌", e)
    else:
        print("✅ All constraints satisfied!")