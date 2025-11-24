
import pandas as pd
import datetime as dt
import ast

def preprocess_data(df_werknemers: pd.DataFrame, df_rooster_template: pd.DataFrame, df_onb: pd.DataFrame, prev_assignments: pd.DataFrame, df_vastrooster: pd.DataFrame, num_weeks: int = 4):
    """
    Prepares shifts and workers dataframes for the OR-Tools scheduling model.
    Returns processed shifts, workers, and helper structures.
    """
    
    print("=== PREPROCESSING DATA ===")    
    # --- Shifts ---
    # Loading from rooster_template
    # Standardize column names for convenience
    df_rooster_template.columns = [
        "Shifts", "actie", "Begintijd", "Eindtijd", "Deskundigheid",
        "Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag",
        "Zaterdag", "Zondag"
    ]

    # Keep only rows where actie == 'plannen'
    df_shifts = df_rooster_template[df_rooster_template["actie"].str.lower() == "plannen"].copy()
    df_shifts = df_shifts.reset_index(drop=True)

    # Convert "Ja"/"Facultatief"/"Nee" → 1/0.5/0
    day_cols = ["Maandag","Dinsdag","Woensdag","Donderdag","Vrijdag","Zaterdag","Zondag"]
    def convert_day_value(x):
        if str(x).strip().lower() == "ja":
            return 1
        elif str(x).strip().lower() == "facultatief":
            return 0.5
        else:
            return 0
    for col in day_cols:
        df_shifts[col] = df_shifts[col].apply(convert_day_value)
    # Convert time strings into datetime.time
    df_shifts["Begintijd"] = pd.to_datetime(df_shifts["Begintijd"], format="%H:%M:%S").dt.time
    df_shifts["Eindtijd"] = pd.to_datetime(df_shifts["Eindtijd"], format="%H:%M:%S").dt.time

    # Convert deskundigheid field: extract numeric code
    # Example: "2. Verzorgende" → 2
    df_shifts["Deskundigheid"] = df_shifts["Deskundigheid"].astype(str).str.extract(r"(\d+)").astype(int)

    # Convert to list format (your code expects: [2], or sometimes multiple)
    df_shifts["Deskundigheid"] = df_shifts["Deskundigheid"].apply(lambda x: [x])

    # Compute duration in hours
    df_shifts["Duur"] = (
        pd.to_datetime(df_shifts["Eindtijd"].astype(str)) -
        pd.to_datetime(df_shifts["Begintijd"].astype(str))
    ).dt.total_seconds() / 3600

    # Fix negative durations (crosses midnight)
    df_shifts.loc[df_shifts["Duur"] < 0, "Duur"] += 24

    # Convert to long format: each row = one shift on one day
    days = ['Maandag', 'Dinsdag', 'Woensdag', 'Donderdag', 'Vrijdag', 'Zaterdag', 'Zondag']
    day_map = {day: i for i, day in enumerate(days)}  # Map day names to day indices (0–6)

    shift_requirements = []

    for idx, row in df_shifts.iterrows():
        for day in days:
            if row[day] == 1:  # If this shift needs to be filled on this day
                shift_requirements.append({
                    "shift_name": row['Shifts'],
                    "day_of_week": day_map[day],  # 0=Monday, etc.
                    "start_time": row['Begintijd'],
                    "end_time": row['Eindtijd'],
                    "qualification": row['Deskundigheid'],
                    "duration": row['Duur'],
                    "required": 1
                })
            elif row[day] == 0.5:  # Facultatief
                shift_requirements.append({
                    "shift_name": row['Shifts'],
                    "day_of_week": day_map[day],  # 0=Monday, etc.
                    "start_time": row['Begintijd'],
                    "end_time": row['Eindtijd'],
                    "qualification": row['Deskundigheid'],
                    "duration": row['Duur'],
                    "required": 0.5
                })

    ### Previous assignments processing ###
    if prev_assignments is not None and not prev_assignments.empty:
        df = prev_assignments.copy()
        print("Processing previous assignments:")
        print(df.head())

        #drop medewerker id if exists
        if 'Medewerker id' in df.columns:
            df = df.drop(columns=['Medewerker id'])
            
        
        # --- 1. Rename columns from actual schedule to scheduler format ---
        df = df.rename(columns={
            "Mw_id": "employee_id",
            "Datum dienst": "shift_date",
            "Dienst": "shift_name",
            "Dienst starttijd": "start_time",
            "Dienst eindtijd": "end_time"
        })

        # --- 2. Parse dates and times ---
        df["shift_date"] = pd.to_datetime(df["shift_date"], dayfirst=True)
        df["start_time"] = pd.to_datetime(df["start_time"], format="%H:%M").dt.time
        df["end_time"] = pd.to_datetime(df["end_time"], format="%H:%M").dt.time

        # --- 3. Add scheduler fields ---
        df["shift_id"] = range(len(df))
        df['is_night'] = df['start_time'].apply(lambda t: t.hour >= 22 or t.hour < 6)
        df["shift_filled"] = True
        df["qualification"] = pd.NA
        df["deskundigheid"] = pd.NA

        # Duration in minutes
        df["duration_min"] = (
            pd.to_datetime(df["end_time"].astype(str)) -
            pd.to_datetime(df["start_time"].astype(str))
        ).dt.total_seconds() / 60
        # Fix negative durations (crosses midnight)
        df.loc[df["duration_min"] < 0, "duration_min"] += 24 * 60

        # --- 4. Compute day_of_week and absolute_day ---
        df = df.sort_values("shift_date").reset_index(drop=True)
        first_day = df["shift_date"].min().normalize()
        df["day_of_week"] = df["shift_date"].dt.weekday
        df["absolute_day"] = (df["shift_date"].dt.normalize() - first_day).dt.days

        # --- 5. Compute relative week ---
        df["week"] = df["absolute_day"] // 7

        # --- 6. Compute global_week like scheduler ---
        global_start_date = first_day - pd.to_timedelta(first_day.weekday(), unit="D")
        df["global_week"] = ((df["shift_date"].dt.normalize() - global_start_date).dt.days // 7)

        prev_assignments = df

        # --- 7. Set start_date for the next scheduling period ---
        start_date = prev_assignments["shift_date"].max() + dt.timedelta(days=1)
        
    else:
        # If no previous assignments, set start_date to next Monday
        today = pd.Timestamp(dt.datetime.now().date())
        start_date = today + pd.to_timedelta((7 - today.weekday()) % 7, unit="D")
        global_start_date = start_date
    
    
    all_shift_instances = []
    for week in range(num_weeks):
        for row in shift_requirements:
            shift_date = start_date + dt.timedelta(days=week * 7 + row["day_of_week"])
            global_week = (shift_date.normalize() - global_start_date).days // 7 + 1
            all_shift_instances.append({
                **row,
                "week": week + 1,
                "absolute_day": week * 7 + row["day_of_week"],
                "shift_date": shift_date,
                "global_week": global_week
            })

    shifts = pd.DataFrame(all_shift_instances)
    shifts['shift_id'] = shifts.index.astype(int)    
    
    # Convert durations (hours → minutes, integers for CP-SAT)
    shifts['duration_min'] = (shifts['duration'].astype(float) * 60).round().astype(int)

    # Ensure int types
    shifts['week'] = shifts['week'].astype(int)
    shifts['day_of_week'] = shifts['day_of_week'].astype(int)

    # Day key for grouping
    shifts['day_key'] = list(zip(shifts['week'], shifts['day_of_week']))

    # add is_night column based on shift_name (e.g., 'N' is night)
    #shifts['is_night'] = shifts['shift_name'].apply(lambda x: 1 if x == 'N' else 0)
    
    # add is_night based on the difference between start_time and end_time being negative
    # compute per-shift whether end_time < start_time (crosses midnight)
    shifts['is_night'] = (
        pd.to_datetime(shifts['end_time'].astype(str)) -
        pd.to_datetime(shifts['start_time'].astype(str))
    ) < pd.Timedelta(0)
    shifts['is_night'] = shifts['is_night'].astype(bool)
    shifts['shift_id'] = shifts.index.astype(int)
    
    shifts['qualification'] = shifts['qualification'].apply(
        lambda q: ast.literal_eval(q) if isinstance(q, str) else q)
    
    shifts = shifts[['shift_id', 'shift_name', 'shift_date', 'week', 'global_week', 'day_of_week', 'absolute_day', 'start_time', 'end_time', 'duration_min', 'qualification', 'is_night', 'day_key', 'required']]
    
    shifts_constant = shifts.copy()
    # Remove all shifts where shift_name = KOK or FM
    shifts_constant = shifts_constant[shifts_constant['shift_name'].isin(['KOK', 'FM'])]
    shifts_constant = shifts_constant.reset_index(drop=True)
    shifts_constant['shift_id'] = shifts_constant.index.astype(int)
    
    shifts = shifts[~shifts['shift_name'].isin(['KOK', 'FM'])]
    shifts = shifts.reset_index(drop=True)
    shifts['shift_id'] = shifts.index.astype(int)
    
    # if prev_assignments is None, copy shifts to prev_assignments with correct date range (4 weeks before start_date)
    if prev_assignments is None or prev_assignments.empty:
        prev_assignments = shifts.copy()
        # Change date range to 4 weeks before start_date
        prev_assignments = prev_assignments[prev_assignments['shift_date'] < start_date]
        prev_assignments = prev_assignments[prev_assignments['shift_date'] >= start_date - pd.to_timedelta(28, unit='D')]
        #-4 on global_week due to starting 4 weeks earlier
        prev_assignments['global_week'] -= 4
        prev_assignments = prev_assignments.reset_index(drop=True)
        prev_assignments['employee_id'] = pd.NA  # No assignments yet
    
    print('Shift data loaded')
    # --- Workers ---
    workers = df_werknemers.copy().reset_index(drop=True)
    workers['medewerker_id'] = workers['medewerker_id'].astype(str)
    
    # delete all rows where wensen = niet plannen
    workers = workers[workers['wensen'] != 'niet plannen']
    workers = workers.reset_index(drop=True)
    
    #change name of in dienst and uit dienst to contract vanaf and contract tm
    workers = workers.rename(columns={'datum indienst': 'contact vanaf', 'datum uit dienst': 'contract tm', 'deskundigheid.1': 'deskundigheid'})
    
    # Convert contract geldig van, contract tm, and geboortedatum to datetime
    workers['contract_vanaf'] = pd.to_datetime(workers['contact vanaf'], format='%d/%m/%Y')
    workers['contract_tm'] = pd.to_datetime(workers['contract tm'], format='%d/%m/%Y')
    #If contract_tm is NaT, set to 31-12-2099
    workers['contract_tm'] = workers['contract_tm'].fillna(pd.Timestamp('2099-12-31'))
    
    # Drop unnecessary columns
    workers = workers.drop(columns=['contact vanaf', 'contract tm'])
    
    #Convert deskundigheid to list of ints
    lsts = []
    for i in range(len(workers)):
        lst = []
        if workers['deskundigheid'][i] <10:
            lst = [int(workers['deskundigheid'][i])]
        else:
            lst = [int(x) for x in str(int(workers['deskundigheid'][i]))]
        
        lsts.append(lst)
        
    workers['deskundigheid'] = lsts

    workers['deskundigheid'] = workers['deskundigheid'].apply(
        lambda q: ast.literal_eval(q) if isinstance(q, str) else q
    )
    
    # Delete all employees with deskundigheid = 5 or 6
    workers = workers[~workers['deskundigheid'].apply(lambda x: isinstance(x, (list, tuple)) and (5 in x or 6 in x))]
    
    # If deskundigheid contains 7, ensure 3 is also present
    workers['deskundigheid'] = workers['deskundigheid'].apply(lambda x: x + [3] if 7 in x and 3 not in x else x)
        
    # If contract soort = oproep, set contracturen to 0
    workers.loc[workers['contract soort'] == 'oproep', 'contracturen'] = 0
    
    # Normalize contract details
    workers['max_days_per_week'] = workers['max_werkdgn_pw'].fillna(0).astype(int)
    
    # Safely convert contracturen to float, fill NaN with 0 before converting to int
    workers['contract_hours'] = workers['contracturen'].fillna(0).astype(float)
    
    # if contract_hours is 0, fill with 9 * max_days_per_week
    workers.loc[workers['contract_hours'] == 0, 'contract_hours'] = workers['max_days_per_week'] * 9
    
    workers['contract_minutes'] = (workers['contract_hours'] * 60).round().astype(int) 
    # Convert geboortedatum to leeftijd based on current day (round down if birthday not yet occurred this year)
    current_date = dt.datetime.now().date()

    workers['leeftijd'] = workers['geboortedatum'].apply(
        lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day))
    )
        
    def parse_list(x, cast=str):
        if pd.isna(x) or str(x).strip() == '':
            return []
        return [cast(s.strip()) for s in str(x).split(',')]

    workers['voorkeur_dagdelen'] = workers['voorkeur dagdelen (dag, avond, nacht)'].apply(
    lambda x: parse_list(x, str))
    
    # split voorkeur_dagdelen list into three separate columns (safe for missing items)
    workers['voorkeur_dag'] = workers['voorkeur_dagdelen'].apply(
        lambda x: x[0].strip() if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], str) else ''
    )
    workers['voorkeur_avond'] = workers['voorkeur_dagdelen'].apply(
        lambda x: x[1].strip() if isinstance(x, (list, tuple)) and len(x) > 1 and isinstance(x[1], str) else ''
    )
    workers['voorkeur_nacht'] = workers['voorkeur_dagdelen'].apply(
        lambda x: x[2].strip() if isinstance(x, (list, tuple)) and len(x) > 2 and isinstance(x[2], str) else ''
    )
    

    workers['patroon'] = workers['patroon'].apply(
        lambda x: parse_list(x, int))

    workers['achtereenvolgende_diensten'] = workers['achtereenvolgende diensten'].apply(
        lambda x: parse_list(x, int))
    
    # split achtereenvolgende_diensten list into two separate columns named achtereenvolgende_diensten_min and achtereenvolgende_diensten_max (safe for missing items)
    workers['min_achtereenvolgende_diensten'] = workers['achtereenvolgende_diensten'].apply(
        lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else 0
    )
    workers['max_achtereenvolgende_diensten'] = workers['achtereenvolgende_diensten'].apply(
        lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else 0
    )

    workers['rust_na_werkperiode'] = workers['rust na werkperiode'].fillna(0).astype(int)
    
    print('NEW WORKERS COLUMNS:')
    print(workers.columns)
    #print dtypes of workers
    print(workers.dtypes)
    
        
    print('Worker data loaded')
    # Create IDs
    emp_ids = workers['medewerker_id'].tolist()
    #emp_ids = [str(e) for e in emp_ids]
    emp_index = {emp: i for i, emp in enumerate(emp_ids)}
    
    # --- onbeschikbaarheid ---
    df_onb = df_onb.copy().reset_index(drop=True)
    # drop Medewerker id, change Mw_id to Medewerker id
    df_onb = df_onb.drop(columns=['Medewerker id'], errors='ignore')
    df_onb = df_onb.rename(columns={'Mw_id':'Medewerker id'})
    
    # --- Constant schedule integration ---
    if df_vastrooster is not None and not df_vastrooster.empty:
        # Normalize employee_id
        df_vastrooster['medewerker_id'] = df_vastrooster['medewerker_id'].astype(str)
        
        # Map day name
        day_map_const = {'maandag':0,'dinsdag':1,'woensdag':2,'donderdag':3,'vrijdag':4,'zaterdag':5,'zondag':6}
        df_vastrooster['day_of_week'] = df_vastrooster['dag'].map(day_map_const)
        # Compute shift_date
        df_vastrooster['shift_date'] = start_date + pd.to_timedelta((df_vastrooster['weekvolgnr']-1)*7 + df_vastrooster['day_of_week'], unit='D')
        
        # Map to shift_id in shifts_constant
        def find_shift_id(row):
            matches = shifts_constant[(shifts_constant['shift_name']==row['dienst']) & (shifts_constant['day_of_week']==row['day_of_week'])]
            return matches['shift_id'].iloc[0] if not matches.empty else None
        df_vastrooster['shift_id'] = df_vastrooster.apply(find_shift_id, axis=1)
        
        # Append to df_onb
        df_const_unavail = df_vastrooster[['medewerker_id','shift_date','shift_id']].copy()
        df_const_unavail = df_const_unavail.rename(columns={'medewerker_id':'Medewerker id','shift_date':'Datum beschikbaarheid'})
        df_const_unavail['Beschikbaarheid'] = 'Niet beschikbaar (constant schedule)'
        # beschikbaarheid tijd vanaf and t/m set to entire day
        df_const_unavail['Beschikbaarheid tijd vanaf'] = pd.to_datetime('00:00', format='%H:%M').time()
        df_const_unavail['Beschikbaarheid tijd t/m'] = pd.to_datetime('23:59', format='%H:%M').time()
        
        df_onb = pd.concat([df_onb, df_const_unavail[['Medewerker id','Datum beschikbaarheid', 'Beschikbaarheid','Beschikbaarheid tijd vanaf','Beschikbaarheid tijd t/m']]], ignore_index=True)
        
        # Subtract constant schedule hours from contract_minutes
        for emp in df_const_unavail['Medewerker id'].unique():
            shift_ids = df_const_unavail[df_const_unavail['Medewerker id']==emp]['shift_id'].dropna().tolist()
            hours_to_subtract = shifts_constant.loc[shifts_constant['shift_id'].isin(shift_ids),'duration_min'].sum()
            workers.loc[workers['medewerker_id']==emp,'contract_minutes'] -= hours_to_subtract
            
            print(f"Subtracted {hours_to_subtract} minutes from employee {emp} due to constant schedule.")
    
    # --- onbeschikbaarheid ---
    # Convert 'Datum beschikbaarheid' to datetime
    df_onb['Datum'] = pd.to_datetime(df_onb['Datum beschikbaarheid'], format='%Y-%m-%d').dt.date
    df_onb['Beschikbaarheid'] = df_onb['Beschikbaarheid'].fillna('Onbekend')
    def ensure_time(x):
        if isinstance(x, dt.time):
            return x
        try:
            return pd.to_datetime(x, format='%H:%M').time()
        except:
            return None

    df_onb['Beschikbaarheid_tijd_vanaf'] = df_onb['Beschikbaarheid tijd vanaf'].apply(ensure_time)
    df_onb['Beschikbaarheid_tijd_tm'] = df_onb['Beschikbaarheid tijd t/m'].apply(ensure_time)
    
    #df_onb['Beschikbaarheid_tijd_vanaf'] = pd.to_datetime(df_onb['Beschikbaarheid tijd vanaf'], format='%H:%M', errors='coerce').dt.time
    #df_onb['Beschikbaarheid_tijd_tm'] = pd.to_datetime(df_onb['Beschikbaarheid tijd t/m'], format='%H:%M', errors='coerce').dt.time
    df_onb = df_onb[['Medewerker id', 'Datum', 'Beschikbaarheid', 'Beschikbaarheid_tijd_vanaf', 'Beschikbaarheid_tijd_tm']]
    print('Onbeschikbaarheid data loaded')


    # --- Helper dictionaries ---
    dur_min = shifts.set_index('shift_id')['duration_min'].to_dict()

    # Grouping by week/day
    shifts_by_week = {}
    #shifts_by_day = {}
    for _, r in shifts.iterrows():
        s = int(r['shift_id'])
        w = int(r['week'])
        day_k = r['day_key']
        shifts_by_week.setdefault(w, []).append(s)
        #shifts_by_day.setdefault(day_k, []).append(s)

    shifts_by_day = shifts.groupby('absolute_day')['shift_id'].apply(list).to_dict()
    
    weeks = sorted(shifts['week'].unique().tolist())
    
    night_shifts = shifts.loc[shifts['is_night'], 'shift_id'].tolist()
    night_shifts_by_week = {
        w: shifts.loc[(shifts['week'] == w) & (shifts['is_night']), 'shift_id'].tolist() for w in weeks}
    
    # Create clean shifts dataframe
    # Delete all rows where shifts = KOK or FM
    shifts = shifts[~shifts['shift_name'].isin(['KOK', 'FM'])]
    shifts = shifts.reset_index(drop=True)
    shifts['shift_id'] = shifts.index.astype(int)

    return {
        "shifts": shifts,
        "workers": workers,
        "onb": df_onb,
        "emp_ids": emp_ids,
        "emp_index": emp_index,
        "dur_min": dur_min,
        "shifts_by_week": shifts_by_week,
        "shifts_by_day": shifts_by_day,
        "weeks": weeks,
        "night_shifts": night_shifts,
        "night_shifts_by_week": night_shifts_by_week,
        "prev_assignments": prev_assignments
    }