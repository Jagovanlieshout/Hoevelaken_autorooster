
import pandas as pd
from ortools.sat.python import cp_model
import datetime as dt

def auto_rooster_standardized(data, time_limit_s=60):
    """
    Main function to create an automatic schedule using OR-Tools CP-SAT solver.
    Expects preprocessed data dictionary with keys:
    - shifts: DataFrame of shifts to be filled
    - workers: DataFrame of workers with their attributes
    - onb: DataFrame of unavailable times
    - emp_ids: List of employee IDs
    - dur_min: Dict mapping shift_id to duration in minutes
    - shifts_by_week: Dict mapping week number to list of shift_ids
    - weeks: List of week numbers in the planning horizon
    - night_shifts: List of shift_ids that are night shifts
    - night_shifts_by_week: Dict mapping week number to list of night shift_ids
    - prev_assignments: DataFrame of previous assignments (can be empty)
    
    Returns a dictionary with:
    - assignments_df: DataFrame of shift assignments
    - all_assignments_df: DataFrame of all assignments including previous
    - uncovered_shifts: List of shift_ids that could not be covered
    - objective_value: Objective value of the solution
    - solver_status: Status of the solver (OPTIMAL, FEASIBLE, etc.)
    """
    
    shifts = data['shifts'].copy()
    workers = data['workers'].copy()
    onb = data['onb'].copy()
    emp_ids = data['emp_ids']
    dur_min = data['dur_min']
    shifts_by_week = data['shifts_by_week']
    weeks = data['weeks']
    night_shifts = data['night_shifts']
    night_shifts_by_week = data['night_shifts_by_week']
    prev_assignments = data['prev_assignments']
    

    if prev_assignments is None:
        prev_assignments = pd.DataFrame(columns=[
            'shift_id', 'shift_date', 'employee_id', 'is_night', 'absolute_day', 'week'
        ])
    
    print("=== AUTO ROOSTER START ===")
    print(f"Planning horizon: weeks {weeks[0]}–{weeks[-1]} ({len(weeks)} weeks)")
    print(f"start_date: {shifts['shift_date'].min().date()}, end_date: {shifts['shift_date'].max().date()}")
    print(f"Shifts: {len(shifts)}, Workers: {len(workers)}")

    # make sure shift_date is datetime (date or Timestamp)
    shifts['shift_date'] = pd.to_datetime(shifts['shift_date']).dt.normalize()  # midnight timestamps
    prev_assignments['shift_date'] = pd.to_datetime(prev_assignments['shift_date']).dt.normalize()

    # helper lists and maps using dates
    dates_list = sorted(shifts['shift_date'].dt.date.unique().tolist())  # list of date objects
    num_weeks = len(weeks)

    # shifts_by_date: date -> [shift_id, ...]
    shifts_by_date = {}
    for _, r in shifts.iterrows():
        s = int(r['shift_id'])
        d = pd.to_datetime(r['shift_date']).date()
        shifts_by_date.setdefault(d, []).append(s)

    # night_shifts_by_date: date -> [night_shift_id,...]
    night_shifts_by_date = {}
    for _, r in shifts[shifts['is_night']].iterrows():
        s = int(r['shift_id'])
        d = pd.to_datetime(r['shift_date']).date()
        night_shifts_by_date.setdefault(d, []).append(s)
        
    #shift type mapping
    shift_type_map = {}
    for _, r in shifts.iterrows():
        sid = int(r['shift_id'])
        shift_name = r['shift_name']
        if 'D' in shift_name:
            shift_type_map[sid] = 'D'
        elif 'A' in shift_name:
            shift_type_map[sid] = 'A'
        elif 'N' in shift_name:
            shift_type_map[sid] = 'N'
        else:
            shift_type_map[sid] = 'Other'


    # helper to get previous consecutive block using dates (returns list of dates)
    def get_last_consecutive_block_dates(prev_assignments_df, emp, is_night=False):
        df = prev_assignments_df[prev_assignments_df['employee_id'] == emp].copy()
        if is_night:
            df = df[df['is_night'] == True]
        if df.empty:
            return []
        # normalized dates sorted ascending
        days = sorted(pd.to_datetime(df['shift_date']).dt.date.unique().tolist())
        # walk from end backwards to collect consecutive tail
        block = []
        for d in reversed(days):
            if not block:
                block.append(d)
            else:
                if (pd.to_datetime(block[-1]) - pd.to_datetime(d)).days == 1:
                    block.append(d)
                else:
                    break
        return list(reversed(block))

    model = cp_model.CpModel()

    # Decision variables: x[(shift_id, emp)]
    x = {(s, emp): model.NewBoolVar(f"x_s{s}_e{emp}")
         for s in shifts['shift_id'] for emp in emp_ids}

    # uncovered
    u = {s: model.NewBoolVar(f"uncovered_s{s}") for s in shifts['shift_id']}

    # n_emp_date: 1 iff employee emp works a night on date d (date object)
    n_emp_date = {}
    for emp in emp_ids:
        for d in dates_list:
            b = model.NewBoolVar(f"n_emp{emp}_d{d.isoformat()}")
            n_emp_date[(emp, d)] = b
            night_ids = night_shifts_by_date.get(d, [])
            if night_ids:
                # if b then sum >=1, else sum == 0
                model.Add(sum(x[(s, emp)] for s in night_ids) >= 1).OnlyEnforceIf(b)
                model.Add(sum(x[(s, emp)] for s in night_ids) == 0).OnlyEnforceIf(b.Not())
            else:
                model.Add(b == 0)

    ### Constraints ###

    # 1) Coverage
    for s in shifts['shift_id']:
        model.Add(sum(x[(s, emp)] for emp in emp_ids) + u[s] == 1)

    # 2) At most one shift per employee per calendar day (use shifts_by_date)
    for emp in emp_ids:
        for d, s_list in shifts_by_date.items():
            if s_list:
                model.AddAtMostOne(x[(s, emp)] for s in s_list)

    # 3) After night shift, no day/evening next day (but night allowed next day)
    # For each emp and each date d: if emp works any night on d then they cannot work non-night shifts on d+1
    for emp in emp_ids:
        for d in dates_list:
            next_d = (pd.to_datetime(d) + pd.Timedelta(days=1)).date()
            if next_d not in shifts_by_date:
                continue
            night_ids_today = night_shifts_by_date.get(d, [])
            next_day_non_night_ids = [s for s in shifts_by_date.get(next_d, []) if not bool(shifts.loc[shifts['shift_id'] == s, 'is_night'].iloc[0])]
            if not night_ids_today or not next_day_non_night_ids:
                continue
            # sum(x_night_today) + sum(x_non_night_nextday) <= len(night_ids_today)
            # if any night worked today -> none of next_day_non_night_ids can be worked.
            model.Add(
                sum(x[(s, emp)] for s in night_ids_today) +
                sum(x[(s, emp)] for s in next_day_non_night_ids)
                <= len(night_ids_today)
            )

    # 4) Max work days per week (kept weekly using shifts_by_week)
    for emp in emp_ids:
        max_days = int(workers.loc[workers['medewerker_id'] == emp, 'max_days_per_week'].iloc[0])
        for w in weeks:
            s_list = shifts_by_week.get(w, [])
            if s_list:
                model.Add(sum(x[(s, emp)] for s in s_list) <= max_days)

    # 5) Contract hours averaged across horizon
    for emp in emp_ids:
        cap_minutes = int(workers.loc[workers['medewerker_id'] == emp, 'contract_minutes'].iloc[0])
        total_shifts = list(shifts['shift_id'].tolist())
        model.Add(sum(dur_min[s] * x[(s, emp)] for s in total_shifts) <= cap_minutes * num_weeks)

    # 6) Respect unavailable times (date-aware)
    # assume onb['Datum'] is a date or datetime; times in 'Beschikbaarheid_tijd_vanaf' and '_tm' are time-like
    for _, r in onb.iterrows():
        emp = r['Medewerker id']
        if emp not in emp_ids:
            continue
        date_unb = pd.to_datetime(r['Datum']).date()
        besch = str(r['Beschikbaarheid']).lower()
        if besch == 'beschikbaar':
            continue
        start_unb = r.get('Beschikbaarheid_tijd_vanaf', None)
        end_unb = r.get('Beschikbaarheid_tijd_tm', None)
        # if times are strings, try to parse to time
        if isinstance(start_unb, str):
            start_unb = pd.to_datetime(start_unb).time()
        if isinstance(end_unb, str):
            end_unb = pd.to_datetime(end_unb).time()
        shifts_that_day = shifts_by_date.get(date_unb, [])
        for s in shifts_that_day:
            shift_start = shifts.loc[shifts['shift_id'] == s, 'start_time'].iloc[0]
            shift_end = shifts.loc[shifts['shift_id'] == s, 'end_time'].iloc[0]
            # overlap check
            if start_unb is not None and end_unb is not None:
                overlap = not (shift_end <= start_unb or shift_start >= end_unb)
            else:
                # if no times provided, treat as fully unavailable that day
                overlap = True
            if overlap:
                model.Add(x[(s, emp)] == 0)

    # 7.1) Max consecutive nights (consider prev_assignments)
    for emp in emp_ids:
        max_consec = 7 if emp in {'602859-1'} else 5  # your CAO exemption set uses strings in your code earlier
        # previous night dates as date objects
        prev_night_dates = get_last_consecutive_block_dates(prev_assignments, emp, is_night=True)
        # current nights dates (unique sorted)
        curr_night_dates = sorted(night_shifts_by_date.keys())
        combined = sorted(set(prev_night_dates + curr_night_dates), key=lambda d: pd.to_datetime(d))
        # slide window of size max_consec + 1 over combined dates and detect strictly consecutive windows
        for i in range(len(combined) - max_consec):
            window = combined[i:i + max_consec + 1]
            # check if strictly consecutive by date difference
            if (pd.to_datetime(window[-1]) - pd.to_datetime(window[0])).days == max_consec:
                # identify which dates in window are in *current horizon*
                window_curr_dates = [d for d in window if d in curr_night_dates]
                if not window_curr_dates:
                    continue
                # find all night shift ids in the current horizon that fall on those dates
                night_shift_ids_in_window = [s for d in window_curr_dates for s in night_shifts_by_date.get(d, [])]
                # enforce at most max_consec nights in that window
                model.Add(sum(x[(s, emp)] for s in night_shift_ids_in_window) <= max_consec)

    # 7.2) After >=3 consecutive nights => 46h rest (2 calendar days)
    # Build boolean n_emp_date already created. Use OnlyEnforceIf with cond_lits.
    for emp in emp_ids:
        # handle prev_assignments tail: if the last prev block >= 3 and ends on day Dprev,
        # then block Dprev+1 and Dprev+2 (if in current horizon)
        prev_nights = get_last_consecutive_block_dates(prev_assignments, emp, is_night=True)
        if len(prev_nights) >= 3:
            dprev = prev_nights[-1]
            for k in [1, 2]:
                block_d = (pd.to_datetime(dprev) + pd.Timedelta(days=k)).date()
                if block_d in dates_list:
                    # block all shifts that day for this employee
                    for s in shifts_by_date.get(block_d, []):
                        model.Add(x[(s, emp)] == 0)

        # now enforce within horizon using OnlyEnforceIf with n_emp_date variables
        for idx in range(2, len(dates_list)):
            d2 = dates_list[idx]
            d1 = dates_list[idx - 1]
            d0 = dates_list[idx - 2]
            # need d2+1 to exist to form condition "not n(d2+1)"
            d2p1 = (pd.to_datetime(d2) + pd.Timedelta(days=1)).date()
            if d2p1 not in n_emp_date and d2p1 not in dates_list:
                continue
            # cond literals: n(d0) & n(d1) & n(d2) & not n(d2+1)
            cond = [
                n_emp_date[(emp, d0)],
                n_emp_date[(emp, d1)],
                n_emp_date[(emp, d2)]
            ]
            # n(d2+1) might not exist if outside horizon; treat absent as 0; but we only enforce if d2+1 is in horizon
            if d2p1 in dates_list:
                cond.append(n_emp_date[(emp, d2p1)].Not())
            else:
                # if next day not in horizon we treat condition only requiring n(d0)&n(d1)&n(d2)
                # but we won't be able to block shifts in missing dates, so skip
                continue

            # blocked days: d2+1 and d2+2 if they exist in horizon
            blocked_days = []
            for k in [1, 2]:
                bd = (pd.to_datetime(d2) + pd.Timedelta(days=k)).date()
                if bd in dates_list:
                    blocked_days.append(bd)
            # collect blocked shift ids
            blocked_shift_ids = [s for bd in blocked_days for s in shifts_by_date.get(bd, [])]
            for bs in blocked_shift_ids:
                model.Add(x[(bs, emp)] == 0).OnlyEnforceIf(cond)

    # 7.3) Max 35 nights per 13 weeks (count prev nights in sliding windows)
    for emp in emp_ids:
        # get previous night shift dates (not necessarily consecutive)
        prev_nights_all = prev_assignments[(prev_assignments['employee_id'] == emp) & (prev_assignments['is_night'] == True)].copy()
        prev_night_shift_ids = prev_nights_all['shift_id'].tolist()

        # for each 13-week window based on your weeks list
        min_week = min(weeks)
        max_week = max(weeks)
        for start_week in range(min_week, max_week - 12 + 1):
            window_weeks = list(range(start_week, start_week + 13))
            window_shifts = [s for w in window_weeks for s in night_shifts_by_week.get(w, [])]
            if not window_shifts:
                continue
            # count how many previous-night shifts fall in these weeks
            prev_count = 0
            for s_prev in prev_night_shift_ids:
                # need mapping shift_id -> week for prev shift (assumed same calendar mapping as shifts)
                try:
                    w_prev = int(prev_assignments.loc[prev_assignments['shift_id'] == s_prev, 'week'].iloc[0])
                    if w_prev in window_weeks:
                        prev_count += 1
                except Exception:
                    # if prev doesn't contain week, skip (or you could compute from date)
                    pass
            model.Add(sum(x[(s, emp)] for s in window_shifts) + prev_count <= 35)

    # 7.4) Age > 55: no night shifts (respect exemptions)
    #CAO_7_4_exempt = {'2097-3', '602859-1'}
    #CAO_7_4_exempt = workers[workers['nachten'] == 'uitsluitend']['medewerker_id'].astype(str).tolist()
    # If nachten column is anything else than 'niet', put worker in CAO_7_4_exempt
    CAO_7_4_exempt = workers[workers['nachten'] != 'niet']['medewerker_id'].astype(str).tolist()
    
    
    for emp in emp_ids:
        if emp in CAO_7_4_exempt:
            continue
        leeftijd = int(workers.loc[workers['medewerker_id'] == emp, 'leeftijd'].iloc[0])
        if leeftijd >= 55:
            for s in night_shifts:
                model.Add(x[(s, emp)] == 0)

    # 8) Deskundigheid rule
    for _, shift_row in shifts.iterrows():
        sid = int(shift_row['shift_id'])

        # required qualification → take minimum qualification level for shift
        req_quals = shift_row['qualification']
        if isinstance(req_quals, list):
            req_level = min(req_quals)   # e.g., [2,3] → requires at least level 2
        else:
            req_level = int(req_quals)

        for emp in emp_ids:
            emp_level = workers.loc[workers['medewerker_id'] == emp, 'deskundigheid'].iloc[0]
            # employee can only work shift if emp_level <= req_level
            if min(emp_level) > req_level:
                model.Add(x[(sid, emp)] == 0)
    
    # 9) Use 'nachten' column to enforce the night shift preferences
    for emp in emp_ids:
        night_emp = workers.loc[workers['medewerker_id'] == emp, 'nachten'].iloc[0]
        if night_emp == 'Niet':
            for s in night_shifts:
                model.Add(x[(s, emp)] == 0)
        elif night_emp == 'uitsluitend':
            for s in shifts['shift_id']:
                if s not in night_shifts:
                    model.Add(x[(s, emp)] == 0)
        

    # ---------- Persoonlijke wensen ----------

    # 10.1) (2097) only nights
    emp_2097 = '2097'
    if emp_2097 in emp_ids:
        for s in shifts['shift_id']:
            if s not in night_shifts:
                model.Add(x[(s, emp_2097)] == 0)

    # 10.2) (2653) only weekends -> use day_of_week from shift row
    emp_2653 = '2653'
    if emp_2653 in emp_ids:
        for _, r in shifts.iterrows():
            s = int(r['shift_id'])
            dow = int(r['day_of_week'])
            if dow not in [5, 6]:
                model.Add(x[(s, emp_2653)] == 0)

    # 10.3) (3888): Friday evening (shift_name has 'A') or weekends
    emp_3888 = '3888'
    if emp_3888 in emp_ids:
        for _, r in shifts.iterrows():
            s = int(r['shift_id'])
            dow = int(r['day_of_week'])
            shift_name = r['shift_name']
            if not ((dow == 4 and 'A' in shift_name) or dow in [5, 6]):
                model.Add(x[(s, emp_3888)] == 0)

    # 10.4) (601011): No Mon/Tue/Wed
    emp_601011 = '601011'
    if emp_601011 in emp_ids:
        for _, r in shifts.iterrows():
            s = int(r['shift_id'])
            dow = int(r['day_of_week'])
            if dow in [0, 1, 2]:
                model.Add(x[(s, emp_601011)] == 0)

    # 10.5) (603722): max 2 days in row and 2 days off after block, with prev_assignments considered
    emp_603722 = '603722-1'
    if emp_603722 in emp_ids:
        # day-level bools keyed by calendar date
        work_day = {}
        for d in dates_list:
            w = model.NewBoolVar(f"work_{emp_603722}_{d.isoformat()}")
            work_day[d] = w
            shifts_today = shifts_by_date.get(d, [])
            if shifts_today:
                model.Add(sum(x[(s, emp_603722)] for s in shifts_today) >= 1).OnlyEnforceIf(w)
                model.Add(sum(x[(s, emp_603722)] for s in shifts_today) == 0).OnlyEnforceIf(w.Not())
            else:
                model.Add(w == 0)

        # incorporate previous tail block length (dates)
        prev_block = get_last_consecutive_block_dates(prev_assignments, emp_603722, is_night=False)
        prev_len = len(prev_block)

        # Max 2 days in a row: for every triple d,d+1,d+2 -> sum <= 2
        for i in range(len(dates_list) - 2):
            d = dates_list[i]
            d1 = dates_list[i + 1]
            d2 = dates_list[i + 2]
            # If prev_len > 0 and i == 0, then we must account previous consecutive tail that may combine with d,d1,d2
            # Use linearization: offset = prev_len but clamp to 2 because prev_len could be >2 which is already infeasible
            offset = min(prev_len, 2) if i == 0 else 0
            # model: work(d)+work(d1)+work(d2) + offset <= 2
            model.Add(work_day[d] + work_day[d1] + work_day[d2] + offset <= 2)

        # 2 days off after block end: handle previous block that ended just before horizon
        if prev_len >= 1:
            # if prev block length >=1 and ended right before start date,
            # we must ensure the first (2 - (prev_len-?)) days are off depending on whether prev block ended >=2... simpler: if prev_len>=2 then first day must be off; if prev_len>=1 then enforce first two days off if block ended two days before? 
            # Simpler robust rule: if prev_len >= 2 then the first day must be off; if prev_len >=3 then first two days must be off.
            if prev_len >= 3:
                for d in dates_list[:2]:
                    model.Add(work_day[d] == 0)
            elif prev_len == 2:
                # require first day off
                model.Add(work_day[dates_list[0]] == 0)
            # if prev_len == 1, we don't force offs (block hasn't reached 2)
        # Also enforce "if works today and NOT tomorrow -> day after tomorrow must be off"
        for i in range(len(dates_list) - 2):
            d = dates_list[i]
            d1 = dates_list[i + 1]
            d2 = dates_list[i + 2]
            model.Add(work_day[d2] == 0).OnlyEnforceIf([work_day[d], work_day[d1].Not()])

    # 10.6) (602859-1): 7-on/7-off pattern using dates, respect prev_assignments
    emp_602859 = '602859-1'
    if emp_602859 in emp_ids:
        print(f'Applying 7-on/7-off pattern for employee {emp_602859}')
        # 1) forbid non-night shifts
        for s in shifts['shift_id']:
            if s not in night_shifts:
                model.Add(x[(s, emp_602859)] == 0)
        # 2) day-level night bools (already have n_emp_date)
        night_work_day = {d: n_emp_date[(emp_602859, d)] for d in dates_list}

        # find last consecutive night block in prev_assignments (dates)
        prev_nights = get_last_consecutive_block_dates(prev_assignments, emp_602859, is_night=True)
        period = 14
        print(f'Previous consecutive night block for {emp_602859}: {prev_nights}')
        # compute offset if we have prior info that constitutes a consistent phase
        prev_phase_offset = None
        if prev_nights:
            # if prev tail has at least one day, we can compute candidate offset: (last_day_index mod 14)
            # choose earliest prev_night for offset calculation
            # offset = (date_index_of_prev_first - date0_index) % 14 where date_index_of_prev_first relates to a continuous day index
            # simpler: take numeric days since reference date0
            try:
                base = pd.to_datetime(dates_list[0])
                prev_first = pd.to_datetime(prev_nights[0])
                print(f'Computing prev_phase_offset for {emp_602859} using base {base.date()} and prev_first {prev_first.date()}')
                prev_phase_offset = (prev_first - base).days % period
                print(f'Computed prev_phase_offset for {emp_602859}: {prev_phase_offset}')
            except Exception:
                prev_phase_offset = None

        # unavailable dates for this emp
        unavailable_dates = set()
        onb_emp = onb[onb['Medewerker id'] == emp_602859]
        for _, r in onb_emp.iterrows():
            date_unb = pd.to_datetime(r['Datum']).date()
            besch = str(r['Beschikbaarheid']).lower()
            if besch != 'beschikbaar':
                unavailable_dates.add(date_unb)
        if prev_phase_offset is not None:
            # enforce exact pattern consistent with prev_phase_offset
            for d in dates_list:
                pos = ((pd.to_datetime(d) - pd.to_datetime(dates_list[0])).days - prev_phase_offset) % period
                print(f'Date {d}: pos {pos} for emp {emp_602859}')
                if pos < 7:
                    # check if date is in unavailable dates
                    if d in unavailable_dates:
                        model.Add(night_work_day[d] == 0)
                    else:
                        model.Add(night_work_day[d] == 1)
                else:
                    model.Add(night_work_day[d] == 0)
        else:
            # allow solver to choose a phase
            offsets = [model.NewBoolVar(f"teuna_phase_{k}") for k in range(period)]
            model.Add(sum(offsets) == 1)
            for k, vk in enumerate(offsets):
                for d in dates_list:
                    pos = ((pd.to_datetime(d) - pd.to_datetime(dates_list[0])).days - k) % period
                    if pos < 7:
                        model.Add(night_work_day[d] == 1).OnlyEnforceIf(vk)
                    else:
                        model.Add(night_work_day[d] == 0).OnlyEnforceIf(vk)

    # 10.8) (602056): max 3 shifts / week, only A or N
    emp_602056 = '602056'
    if emp_602056 in emp_ids:
        for w in weeks:
            s_list = shifts_by_week.get(w, [])
            if s_list:
                model.Add(sum(x[(s, emp_602056)] for s in s_list) <= 3)
                for s in s_list:
                    shift_name = shifts.loc[shifts['shift_id'] == s, 'shift_name'].iloc[0]
                    if 'A' not in shift_name and 'N' not in shift_name:
                        model.Add(x[(s, emp_602056)] == 0)
                        
    # Standardized constraints
    
    # 11.1) pattern constraint for employees with non empty 'patroon' field
    for emp in emp_ids:
        patroon = str(workers.loc[workers['medewerker_id'] == emp, 'patroon'].iloc[0]).strip().lower()
        night_emp = str(workers.loc[workers['medewerker_id'] == emp, 'nachten'].iloc[0]).strip().lower()
        
        if patroon and patroon != 'nan':
            print(f'Applying pattern constraint for employee {emp} with pattern {patroon}')
        else:
            continue
            
        if night_emp == 'uitsluitend':
            night = True
        else:
            night = False
        pattern_length = int(float(patroon))
        print(f'Pattern length for employee {emp}: {pattern_length} using {night} nightshifts')
        
                        
    ### Objective ###
    
    # 1) Uncovered shifts penalties with lower weight for D4/A3
    uncovered_terms = []
    for _, r in shifts.iterrows():
        sid = int(r['shift_id'])
        weight = 0.5 if r['shift_name'] in ['D4', 'A3'] else 1.0
        uncovered_terms.append(weight * u[sid])

    # 2) Under-coverage penalties for non weekend workers
    under_coverage_terms = []
    under_coverage_weekend_terms = []
    wensen_map = {emp: str(workers.loc[workers['medewerker_id'] == emp, 'wensen']).lower() for emp in emp_ids}
    employees_no_weekend_pref = [e for e in emp_ids if 'weekend' not in wensen_map[e]]
    employees_weekend_pref = [e for e in emp_ids if 'weekend' in wensen_map[e]]      
    for emp in employees_no_weekend_pref:
        cap_minutes = int(workers.loc[workers['medewerker_id'] == emp, 'contract_minutes'].iloc[0])
        total_shifts = list(shifts['shift_id'].tolist())
        under_coverage = model.NewIntVar(0, cap_minutes * num_weeks, f"under_coverage_e{emp}")
        squared_under_coverage = model.NewIntVar(0, cap_minutes * cap_minutes * num_weeks * num_weeks, f"squared_under_coverage_e{emp}")
        model.Add(sum(dur_min[s] * x[(s, emp)] for s in total_shifts) + under_coverage >= cap_minutes * num_weeks)        
        #under_coverage_terms.append(under_coverage)
        under_coverage_terms.append(squared_under_coverage)  
 

    for emp in employees_weekend_pref:
        cap_minutes = int(workers.loc[workers['medewerker_id'] == emp, 'contract_minutes'].iloc[0])
        total_shifts = list(shifts['shift_id'].tolist())
        under_coverage = model.NewIntVar(0, cap_minutes * num_weeks, f"under_coverage_e{emp}")
        squared_under_coverage = model.NewIntVar(0, cap_minutes * cap_minutes * num_weeks * num_weeks, f"squared_under_coverage_e{emp}")
        model.Add(sum(dur_min[s] * x[(s, emp)] for s in total_shifts) + under_coverage >= cap_minutes * num_weeks)        
        #under_coverage_weekend_terms.append(under_coverage)
        under_coverage_weekend_terms.append(squared_under_coverage)

        
    
    # 3) Penalty for consecutive weekends worked for fixed-contract workers
    # weekendWorked[(emp, week)] = 1 if employee emp works any shift on Saturday(5) or Sunday(6) in that week
    weekendWorked = {}
    weekend_days = [5, 6]  # day_of_week indices for Saturday and Sunday
    

    # Precompute weekend shift ids per week for speed
    weekend_shifts_by_week = {}
    for w in weeks:
        weekend_shifts = shifts.loc[(shifts['week'] == w) & (shifts['day_of_week'].isin(weekend_days)), 'shift_id'].tolist()
        weekend_shifts_by_week[w] = weekend_shifts

    # Only consider employees with contract soort == 'vaste uren'
    #vaste_uren_employees = [e for e in emp_ids if workers.loc[workers['medewerker_id'] == e, 'contract soort'].iloc[0] == 'vaste uren']
    
    for emp in employees_no_weekend_pref:
        for w in weeks:
            w_shifts = weekend_shifts_by_week.get(w, [])
            var = model.NewBoolVar(f"weekendWorked_e{emp}_w{w}")
            weekendWorked[(emp, w)] = var
            if w_shifts:
                # If var True => sum(x[s,emp]) >= 1 ; if var False => sum == 0
                model.Add(sum(x[(s, emp)] for s in w_shifts) >= 1).OnlyEnforceIf(var)
                model.Add(sum(x[(s, emp)] for s in w_shifts) == 0).OnlyEnforceIf(var.Not())
            else:
                # no weekend shifts this week => cannot work weekend
                model.Add(var == 0)

    # Create consec weekend penalty booleans where we can compare week w and w+1
    consec_weekend_penalties = []
    for emp in employees_no_weekend_pref:
        # ensure weeks are sorted; assume 'weeks' already sorted in preprocessing
        for i in range(len(weeks) - 1):
            w = weeks[i]
            w_next = weeks[i + 1]
            # Only create penalty var if both weeks exist in horizon
            consec = model.NewBoolVar(f"consecWeekend_e{emp}_w{w}")
            # Link consec to the AND of weekendWorked[(emp,w)] and weekendWorked[(emp,w_next)]
            # If consec is true then both weekendWorked must be true
            model.AddBoolAnd([weekendWorked[(emp, w)], weekendWorked[(emp, w_next)]]).OnlyEnforceIf(consec)
            # If consec is false then at least one of the weekendWorked is false
            model.AddBoolOr([weekendWorked[(emp, w)].Not(), weekendWorked[(emp, w_next)].Not()]).OnlyEnforceIf(consec.Not())
            consec_weekend_penalties.append(consec)
    
    # --- Add check with previous schedule  ---
    for emp in employees_no_weekend_pref:
        # find last weekend in previous schedule (if any)
        prev_weekend_dates = prev_assignments[
            (prev_assignments['employee_id'] == emp) &
            (prev_assignments['shift_date'].dt.dayofweek.isin(weekend_days))]['shift_date'].dt.date.unique()
        worked_last_prev_weekend = len(prev_weekend_dates) > 0
        
        if worked_last_prev_weekend and weekend_shifts_by_week.get(weeks[0]):            # last weekend date in previous schedule
            consec_prev = model.NewBoolVar(f"consecWeekend_prev_e{emp}")
            model.AddBoolAnd([weekendWorked[(emp, weeks[0])]]).OnlyEnforceIf(consec_prev)
            model.AddBoolOr([weekendWorked[(emp, weeks[0])].Not()]).OnlyEnforceIf(consec_prev.Not())
            consec_weekend_penalties.append(consec_prev) 

    # 4) Penalty for isolated shifts
    # Create day-level work variables: work_day[(emp, date)] = 1 if employee works any shift on that date
    work_day = {}
    for emp in emp_ids:
        for d in dates_list:
            shifts_today = shifts_by_date.get(d, [])
            var = model.NewBoolVar(f"workDay_e{emp}_d{d.isoformat()}")
            work_day[(emp, d)] = var
            if shifts_today:
                # If var True => works at least one shift; If var False => works 0 shifts
                model.Add(sum(x[(s, emp)] for s in shifts_today) >= 1).OnlyEnforceIf(var)
                model.Add(sum(x[(s, emp)] for s in shifts_today) == 0).OnlyEnforceIf(var.Not())
            else:
                # No shifts on this date => cannot work
                model.Add(var == 0)

    # Precompute last day worked from previous schedule
    last_prev_day_worked = {}
    for emp in emp_ids:
        prev_emp_shifts = prev_assignments[prev_assignments['employee_id'] == emp]
        if not prev_emp_shifts.empty:
            # Get the max date the employee worked
            last_prev_day_worked[emp] = prev_emp_shifts['shift_date'].max().date()
        else:
            last_prev_day_worked[emp] = None
    
    # Create isolated shift penalty variables
    isolated_shift_penalties = []
    for emp in emp_ids:
        for idx, d in enumerate(dates_list):
            # Only consider days with a previous or next day
            prev_d = dates_list[idx - 1] if idx > 0 else None
            next_d = dates_list[idx + 1] if idx < len(dates_list) - 1 else None

            if prev_d is None and next_d is None:
                continue  # Single-day horizon, skip

            # Boolean variable: 1 if shift is isolated
            iso_var = model.NewBoolVar(f"isolatedShift_e{emp}_d{d.isoformat()}")

            conds = [work_day[(emp, d)]]  # Must work on this day
            
            # condition: previous day not worked
            if prev_d is not None:
                conds.append(work_day[(emp, prev_d)].Not())  # Not worked previous day
            else:
                # check previous schedule for first day
                if last_prev_day_worked[emp] is not None and last_prev_day_worked[emp] == (pd.to_datetime(d) - pd.Timedelta(days=1)).date():
                    # employee worked day before first day -> do NOT count as isolated
                    conds = []  # cancel isolation condition
                    iso_var = model.NewBoolVar(f"isolatedShift_e{emp}_d{d.isoformat()}")  # dummy, always 0
                    model.Add(iso_var == 0)
            
            # condition: next day not worked
            if next_d is not None:
                conds.append(work_day[(emp, next_d)].Not())  # Not worked next day

            # iso_var is true <=> employee works this day and no adjacent work
            model.AddBoolAnd(conds).OnlyEnforceIf(iso_var)
            model.AddBoolOr([work_day[(emp, d)].Not()] + ([work_day[(emp, prev_d)]] if prev_d else []) + ([work_day[(emp, next_d)]] if next_d else [])).OnlyEnforceIf(iso_var.Not())

            # Add to list for objective
            isolated_shift_penalties.append(iso_var)

    # 5): insufficient rest after night shift blocks
    rest_after_night_penalties = []
    for emp in emp_ids:
        # Combine previous night shifts with current horizon
        prev_nights = get_last_consecutive_block_dates(prev_assignments, emp, is_night=True)
        curr_nights = sorted(night_shifts_by_date.keys())
        combined_night_dates = sorted(set(prev_nights + curr_nights), key=lambda d: pd.to_datetime(d))

        # slide window over current horizon
        for i, d in enumerate(dates_list):
            # Check if this date is immediately after a night shift block
            if i < 2:
                continue  # first two days cannot have full 2-day rest check

            # find if previous two days were night shifts (part of block)
            d_minus_1 = dates_list[i - 1]
            d_minus_2 = dates_list[i - 2]

            # Boolean: was d_minus_1 or d_minus_2 night shift? (to detect block end)
            if d_minus_2 in combined_night_dates and d_minus_1 in combined_night_dates:
                # if both were nights, still part of block => skip rest check
                continue

            # If yesterday was night shift but today is first day off (or next two days)
            # We penalize if today or tomorrow is worked
            if d_minus_1 in curr_nights:
                for k in range(2):  # check next 2 days
                    if i + k < len(dates_list):
                        d_check = dates_list[i + k]
                        # create penalty variable
                        pen_var = model.NewBoolVar(f"restAfterNight_e{emp}_d{d_check.isoformat()}")
                        # penalize if employee works
                        model.Add(pen_var == 1).OnlyEnforceIf(work_day[(emp, d_check)])
                        model.Add(pen_var == 0).OnlyEnforceIf(work_day[(emp, d_check)].Not())
                        rest_after_night_penalties.append(pen_var)
    
    # 6): Penalty for uneven distribution of type of shift per employee
    shift_type_count = {}
    for emp in emp_ids:
        for t in ['D', 'A', 'N']:
            shift_type_count[(emp, t)] = model.NewIntVar(0, len(shifts), f"shiftCount_{emp}_{t}")

        # Sum assignments for each type
        for t in ['D', 'A', 'N']:
            type_shifts = [s for s in shifts['shift_id'] if shift_type_map[s] == t]
            if type_shifts:
                model.Add(shift_type_count[(emp, t)] == sum(x[(s, emp)] for s in type_shifts))
            else:
                # if no shifts of this type exist, count = 0
                model.Add(shift_type_count[(emp, t)] == 0)

    max_shift_count = {}
    for emp in emp_ids:
        total_shifts = model.NewIntVar(0, len(shifts), f"totalShifts_{emp}")
        model.Add(total_shifts == sum(shift_type_count[(emp, t)] for t in ['D', 'A', 'N']))
        
        max_shift_count[emp] = model.NewIntVar(0, len(shifts), f"maxShiftCount_{emp}")
        model.AddMaxEquality(max_shift_count[emp], [shift_type_count[(emp, t)] for t in ['D', 'A', 'N']])

    unequal_shift_penalties = []
    for emp in emp_ids:
        pen_var = model.NewBoolVar(f"unequalShiftPenalty_{emp}")
        
        # Only apply penalty if employee has at least 1 shift
        model.Add(max_shift_count[emp] * 2 > total_shifts).OnlyEnforceIf(pen_var)
        model.Add(max_shift_count[emp] * 2 <= total_shifts).OnlyEnforceIf(pen_var.Not())
        
        unequal_shift_penalties.append(pen_var)
    
    # 7) Equal distribution of amount of shifts per week per employee
    # Keep out Teuna (602859) from this penalty as she has fixed 7-on/7-off pattern
    equal_dist_emp_ids = [e for e in emp_ids if e not in {'602859-1'}]

    # Count shifts per week per employee
    shifts_per_week = {}
    for emp in equal_dist_emp_ids:
        for w in weeks:
            week_shifts = shifts.loc[shifts['week'] == w, 'shift_id'].tolist()
            shifts_per_week[(emp, w)] = model.NewIntVar(0, len(week_shifts), f"shiftsPerWeek_e{emp}_w{w}")
            if week_shifts:
                model.Add(shifts_per_week[(emp, w)] == sum(x[(s, emp)] for s in week_shifts))
            else:
                model.Add(shifts_per_week[(emp, w)] == 0)

    # Compute total shifts per employee
    total_shifts_emp = {}
    for emp in equal_dist_emp_ids:
        total_shifts_emp[emp] = model.NewIntVar(0, len(shifts), f"totalShifts_e{emp}")
        model.Add(total_shifts_emp[emp] == sum(shifts_per_week[(emp, w)] for w in weeks))

    # Create deviation variables per week (fractional average fairness)
    week_balance_penalties = {}
    num_weeks = len(weeks)

    for emp in equal_dist_emp_ids:
        for w in weeks:
            # expr = week_shifts * num_weeks - total_shifts
            expr = shifts_per_week[(emp, w)] * num_weeks - total_shifts_emp[emp]

            dev_pos = model.NewIntVar(0, len(shifts) * num_weeks, f"weekDevPos_e{emp}_w{w}")
            dev_neg = model.NewIntVar(0, len(shifts) * num_weeks, f"weekDevNeg_e{emp}_w{w}")

            model.Add(expr == dev_pos - dev_neg)
            week_dev = model.NewIntVar(0, len(shifts) * num_weeks, f"weekDevAbs_e{emp}_w{w}")
            model.Add(week_dev == dev_pos + dev_neg)

            # Create a squared penalty using AddMultiplicationEquality
            week_dev_sq = model.NewIntVar(0, (len(shifts) * num_weeks) ** 2, f"weekDevSq_e{emp}_w{w}")
            model.AddMultiplicationEquality(week_dev_sq, [week_dev, week_dev])

            week_balance_penalties[(emp, w)] = week_dev_sq        

    # 8) Use 'nachten' column to penalize employees with 'overig' for each night shift they are assigned to
    overig_night_penalties = []
    for emp in emp_ids:
        night_emp = workers.loc[workers['medewerker_id'] == emp, 'nachten'].iloc[0]
        if night_emp == 'overig':
            for s in night_shifts:
                pen_var = model.NewBoolVar(f"overigNightPenalty_e{emp}_s{s}")
                model.Add(pen_var == x[(s, emp)])
                overig_night_penalties.append(pen_var)
    # 9) Give a small bonus for employees working their preferred shifts
    preferred_shift_bonus = []
    
    for _, r in onb.iterrows():
        emp = r['Medewerker id']
        if emp not in emp_ids:
            continue
        date_unb = pd.to_datetime(r['Datum']).date()
        besch = str(r['Beschikbaarheid']).lower()
        if besch == 'beschikbaar':
            # find shifts on that date
            shifts_on_date = shifts_by_date.get(date_unb, [])
            for s in shifts_on_date:
                # create a bonus variable
                bonus_var = model.NewBoolVar(f"bonus_e{emp}_s{s}")
                model.Add(bonus_var == x[(s, emp)])  # bonus_var = 1 iff employee works shift
                preferred_shift_bonus.append(bonus_var)
                
    # 10) Penalty for deskundigheid level higher than required (to prefer lower levels when possible)
    deskundigheid_penalties = []
    for _, shift_row in shifts.iterrows():
        sid = int(shift_row['shift_id'])

        req_quals = shift_row['qualification']
        req_level = min(req_quals) if isinstance(req_quals, list) else int(req_quals)

        for emp in emp_ids:
            emp_level = workers.loc[workers['medewerker_id'] == emp, 'deskundigheid'].iloc[0]

            # Allowed assignments only (because emp_level ≤ req_level)
            if min(emp_level) <= req_level:
                diff = req_level - max(emp_level)    # e.g. 3 - 1 = 2
                penalty_value = diff * diff      # quadratic

                # Create an integer penalty variable for this assignment
                p = model.NewIntVar(0, penalty_value, f"penalty_s{sid}_e{emp}")

                # Link penalty to assignment:
                # p = penalty_value * x[sid, emp]
                # CP-SAT cannot multiply IntVar * IntVar → linearize with AddMultiplicationEquality
                model.AddMultiplicationEquality(p, [x[(sid, emp)], penalty_value])

                deskundigheid_penalties.append(p)
        
            
    # Combine all into one objective
    model.Minimize(
        10 * sum(uncovered_terms) +                          # main uncovered penalty
        0.005 * sum(under_coverage_terms) +             # soft penalty for under-coverage
        0.001 * sum(under_coverage_weekend_terms) +   # soft penalty for under-coverage weekend pref
        5 * sum(consec_weekend_penalties) +           # soft penalty for consecutive weekends
        1 * sum(isolated_shift_penalties) +           # soft penalty for isolated shifts
        0.5 * sum(rest_after_night_penalties) +         # soft penalty for insufficient rest after night blocks
        0.1 * sum(unequal_shift_penalties) +            # soft penalty for uneven shift type distribution
        0.1 * sum(week_balance_penalties.values()) +   # soft penalty for unequal distribution of shifts per week
        1 * sum(overig_night_penalties) -              # soft penalty for 'overig' night shifts
        0.1 * sum(preferred_shift_bonus) +              # small bonus for preferred shifts
        0.1 * sum(deskundigheid_penalties)                 # soft penalty for higher than required deskundigheid
    )

    ### Solve ###
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8
    try:
        status = solver.Solve(model)
    except Exception as e:
        import traceback
        print("Solver exception caught!")
        traceback.print_exc()
        raise

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments = []
        uncovered = []
        for _, r in shifts.iterrows():
            sid = int(r['shift_id'])
            assigned = False
            for emp in emp_ids:
                if solver.Value(x[(sid, emp)]) == 1:
                    assignments.append({
                        'shift_id': sid,
                        'shift_name': r['shift_name'],
                        'start_time': r['start_time'],
                        'end_time': r['end_time'],
                        'shift_date': r['shift_date'],
                        'is_night': r['is_night'],
                        'week': int(r['week']),
                        'global_week': int(r['global_week']),
                        'day_of_week': int(r['day_of_week']),
                        'absolute_day': int(r['absolute_day']),
                        'duration_min': int(r['duration_min']),
                        'employee_id': str(emp),
                        'employee_name': workers.loc[workers['medewerker_id'] == emp, 'medewerker_naam'].iloc[0],
                        'qualification': r['qualification'],
                        'deskundigheid': workers.loc[workers['medewerker_id'] == emp, 'deskundigheid'].iloc[0],
                        'shift_filled': True
                    })
                    assigned = True
                    break
            if not assigned:
                uncovered.append(sid)
                assignments.append({
                    'shift_id': sid,
                    'shift_name': r['shift_name'],
                    'start_time': r['start_time'],
                    'end_time': r['end_time'],
                    'shift_date': r['shift_date'],
                    'is_night': r['is_night'],
                    'week': int(r['week']),
                    'global_week': int(r['global_week']),
                    'day_of_week': int(r['day_of_week']),
                    'absolute_day': int(r['absolute_day']),
                    'duration_min': int(r['duration_min']),
                    'employee_id': None,
                    'employee_name': None,
                    'qualification': r['qualification'],
                    'deskundigheid': None,
                    'shift_filled': False
                })
        
        # append new assignments to prev_assignments
        all_assignments = pd.concat([prev_assignments, pd.DataFrame(assignments)], ignore_index=True)
        
        print('==== Solution Summary ====')
        print(f"Solution found with objective value {solver.ObjectiveValue()}")
        print(f"Total uncovered shifts: {len(uncovered)}")
        print(f"Total under-coverage penalties: {sum(solver.Value(v) for v in under_coverage_terms)}")
        print("Total consecutive weekend penalties:", sum(solver.Value(v) for v in consec_weekend_penalties))
        print("Total isolated shift penalties:", sum(solver.Value(v) for v in isolated_shift_penalties))
        print("Total insufficient rest after night penalties:", sum(solver.Value(v) for v in rest_after_night_penalties))
        print("Total unequal shift type distribution penalties:", sum(solver.Value(v) for v in unequal_shift_penalties))
        print("Total weekly balance penalties:", sum(solver.Value(v) for v in week_balance_penalties.values()))
        print("Total 'overig' night shift penalties:", sum(solver.Value(v) for v in overig_night_penalties))
        print("Total preferred shift bonuses:", sum(solver.Value(v) for v in preferred_shift_bonus))
        print("Total deskundigheid penalties:", sum(solver.Value(v) for v in deskundigheid_penalties))
        
                
        # ---- Debug print for weekly distribution ----
        print("\n--- Weekly shift distribution (balanced) ---")
        for emp in equal_dist_emp_ids:
            counts = [solver.Value(shifts_per_week[(emp, w)]) for w in weeks]
            total = solver.Value(total_shifts_emp[emp])
            devs = [solver.Value(week_balance_penalties[(emp, w)]) for w in weeks]
            print(f"Employee {emp}: weeks={counts}, total={total}, deviations={devs}, sum_dev={sum(devs)}")
            
        # ---- Debug print for under-coverage ----
        print("\n--- Under-coverage details ---")
        for emp in employees_no_weekend_pref:
            under_cov = solver.Value(under_coverage_terms[employees_no_weekend_pref.index(emp)])
            if under_cov > 0:
                print(f"Employee {emp} ({workers.loc[workers['medewerker_id'] == emp, 'medewerker_naam'].iloc[0]}): under-coverage penalty = {under_cov} minutes")
        
        # ---- Debug print for weekend under-coverage ----
        print("\n--- Weekend preference under-coverage details ---")
        for emp in employees_weekend_pref:
            under_cov = solver.Value(under_coverage_weekend_terms[employees_weekend_pref.index(emp)])
            if under_cov > 0:
                print(f"Employee {emp} ({workers.loc[workers['medewerker_id'] == emp, 'medewerker_naam'].iloc[0]}): weekend pref under-coverage penalty = {under_cov} minutes")
        
        print("All employees with their number of assigned shifts:")
        for emp in emp_ids:
            num_assigned = sum(1 for s in shifts['shift_id'] if solver.Value(x[(s, emp)]) == 1)
            print(f"Employee {emp} ({workers.loc[workers['medewerker_id'] == emp, 'medewerker_naam'].iloc[0]}): assigned shifts = {num_assigned}")
        
        #save to CSV
        assignments_df = pd.DataFrame(assignments)        
        
        return {
            "assignments_df": assignments_df,
            "all_assignments_df": all_assignments,
            "uncovered_shifts": uncovered,
            "objective_value": solver.ObjectiveValue(),
            "solver_status": solver.StatusName(status)
        }
    else:
        print("No solution found:", solver.StatusName(status))
        return None