import pandas as pd
import fastf1 as ff1
from typing import List, Dict, Optional


class F1Session:
    '''
    Class to load and preprocess F1 session data.
    '''

    def __init__(self, data: List[Dict[str, str]]):
        self.data = data
        self.index = 0

    @staticmethod
    def _add_year_circuit(df: pd.DataFrame, year: str, circuit: str) -> pd.DataFrame:
        df['Year'] = year
        df['Circuit'] = circuit
        return df

    def get_lap_times(self, year: str, circuit: str, session_type: str) -> Optional[pd.DataFrame]:
        '''
        Get the lap times with telemetry for a given race.
        '''
        try:
            session = self.load_session(year, circuit, session_type)
            laps = session.laps
            return self._add_year_circuit(laps, year, circuit)

        except KeyError as error:
            print(f"Error retrieving lap times for {year}, {circuit}: {error}")
            return None

    def get_session_results(self, year: str, circuit: str, session_type: str) -> Optional[pd.DataFrame]:
        '''
        Get the results of the session.
        '''
        try:
            session = self.load_session(year, circuit, session_type)
            results = session.results
            return self._add_year_circuit(results, year, circuit)

        except KeyError as error:
            print(
                f"Error retrieving session results for {year}, {circuit}: {error}")
            return None

    def load_session(self, year: str, circuit: str, session_type: str):
        '''
        Load the F1 session.
        '''
        session = ff1.get_session(year, circuit, session_type)
        session.load()
        return session

    @staticmethod
    def preprocess_data(session_results_df: pd.DataFrame, lap_times_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Preprocess the session results and lap times data.
        '''
        rename_dict = {'Abbreviation': 'Driver', 'TeamName': 'Team'}
        session_results_df = session_results_df[[
            'Abbreviation', 'Year', 'Circuit', 'TeamName', 'Position']].rename(columns=rename_dict)

        lap_times_df = lap_times_df[[
            'Driver', 'Year', 'Circuit', 'Team', 'LapTime']]
        lap_times_df['LapTime'] = pd.to_timedelta(
            lap_times_df['LapTime']).dt.total_seconds()

        # group the lap times by driver, year, circuit and team and calculate the mean lap time
        lap_times_df = lap_times_df.groupby(
            ['Driver', 'Year', 'Circuit', 'Team']).mean().reset_index()

        # join the two dataframes on driver, year, circuit and team
        df = pd.merge(session_results_df, lap_times_df, on=[
                      'Driver', 'Year', 'Circuit', 'Team'])

        df['Podium'] = df['Position'].apply(lambda x: 1 if x <= 3 else 0)

        df = df.drop(columns=['Position'])

        return df

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration

        session_data = self.data[self.index]
        year = session_data['year']
        circuit = session_data['circuit']
        session_type = session_data['session_type']
        self.index += 1

        lap_times = self.get_lap_times(year, circuit, session_type)
        session_results = self.get_session_results(year, circuit, session_type)

        if lap_times is None or session_results is None:
            return None

        preprocessed_data = self.preprocess_data(session_results, lap_times)

        return preprocessed_data
