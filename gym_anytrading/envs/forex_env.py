import numpy as np
import pandas as pd
import finta

from .trading_env import TradingEnv, Actions, Positions


class MyForexEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, trade_fee, spread, spread_bool=False, unit_side='right', render_mode=None, **kwargs):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        # get the additional parameters provided in kwargs
        self.kwargs = kwargs

        super().__init__(df, window_size, render_mode)
        # adjust all of the column names to lowercase
        self.df.columns = self.df.columns.str.lower()

        # need to edit this
        self.trade_fee = trade_fee  # unit

        self.spread = spread
        self.spread_bool = spread_bool

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        # It tries to access the element at the index self.frame_bound[0] - self.window_size in the prices array. 
        # If this index is out of bounds, it will raise an IndexError, stopping the execution of the program.
        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        # calculate the sma of the open high low, close / 4 for 3 periods
        self.df.loc[:, 'ohlc4'] = (self.df['close'] + self.df['open'] + self.df['high'] + self.df['low']) / 4
        self.df.loc[:, 'sma'] = self.df['ohlc4'].rolling(window=self.kwargs['sma_length']).mean()
        # calculate the sma of sma3 for 3 periods
        self.df.loc[:, 'smoothing_sma'] = self.df['sma'].rolling(window=self.kwargs['smoothing_sma']).mean()
        # find the difference between sma and smoothing_sma
        self.df.loc[:, 'sma_diff'] = self.df['sma'] - self.df['smoothing_sma']
        # find the sign of the sma_diff
        self.df.loc[:, 'sma_sign'] = np.sign(self.df['sma_diff'])
        # find where the sma_sign changes from 1 to -1 or -1 to 1 or from 1
        self.df.loc[:, 'sma_crossover'] = np.where((self.df['sma_sign'] == 1) & (self.df['sma_sign'].shift(1) == -1), 1, 
                                        np.where((self.df['sma_sign'] == -1) & (self.df['sma_sign'].shift(1) == 1), -1, 
                                        np.where((self.df['sma_sign'] == -1) & (self.df['sma_sign'].shift(1) == 0) & 
                                                 (self.df['sma_sign'].shift(2) == 1), -1,
                                        np.where((self.df['sma_sign'] == 1) & (self.df['sma_sign'].shift(1) == 0) & 
                                                 (self.df['sma_sign'].shift(2) == -1), 1, 0))))
        
        # add the day of the week to the dataframe
        self.df['day_of_week'] = self.df.index.day_name()
        # place a 1 in day_of_week_transition, if it is the last bar on Friday and the next bar is Sunday
        self.df['day_of_week_transition'] = np.where((self.df['day_of_week'] == 'Friday') & 
                                                            ((self.df['day_of_week'].shift(-1) == 'Sunday') | 
                                                            (self.df['day_of_week'].shift(-1) == 'Monday') |
                                                            (self.df['day_of_week'].shift(-1) == 'Tuesday')), 1, 0)
        
        # add the news event to the dataframe
        path_to_news = 'C:/Users/WilliamFetzner/Documents/Trading/calendar_df_full.csv'
        news_df = pd.read_csv(path_to_news)

        if 'Unnamed: 0' in news_df.columns:
            news_df.drop('Unnamed: 0', axis=1, inplace=True)
        # # convert datetime to a datetime object
        news_df.loc[:, 'datetime'] = pd.to_datetime(news_df['datetime'])

        # group the data by the datetime column and count the number of events
        news_df_grouped = news_df.groupby('datetime')['Event'].count()
        # convert caledndar_df_full_grouped to a dataframe
        news_df_grouped = news_df_grouped.to_frame()

        # create a new column in self.df called 'news_event_5' and place a 1 in it if there is an event in the news_df_grouped 
        # dataframe that has a datetime that is within 5 minutes of the datetime in the self.df dataframe
        # first add a new column in self.df called datetime_5 that is the datetime column increased by 5 minutes
        self.df.loc[:, 'datetime_5'] = self.df.index + pd.Timedelta(minutes=5)
        self.df.loc[:, 'datetime_neg_5'] = self.df.index - pd.Timedelta(minutes=5)
        # Step 2: Initialize the new column 'news_event_5' with 0s
        self.df.loc[:, 'news_event_5'] = 0

        # add the minutes until the next news event to the dataframe
        # Initialize the new column 'secs_until_next_news_event' with np.nan
        self.df.loc[:, 'secs_until_next_news_event'] = np.nan
        news_g = news_df_grouped.copy()

        # Step 3: Iterate over each row in self.df and check for events in news_df_grouped
        for index, row in self.df.iterrows():
            # filter the calendar_df_full_grouped dataframe to just the events that are >= the datetime in the self.df dataframe
            news_g = news_g[news_g.index >= index]
                # find the difference between the datetime in the calendar_df_full_grouped dataframe and the index
            # of the self.df dataframe
            news_g['diff'] = news_g.index - index
            # find the minimum difference that is positive
            news_g['diff'] = news_g['diff'].apply(lambda x: x.total_seconds())
            news_g['diff'] = news_g['diff'].apply(lambda x: x if x > 0 else np.nan)
            # find the minimum difference and place it into the secs_until_next_news_event column
            self.df.at[index, 'secs_until_next_news_event'] = news_g['diff'].min()


            # Find events in news_df_grouped that fall within the 5-minute window
            events_in_window = news_df_grouped[(news_df_grouped.index >= row['datetime_neg_5']) & 
                                                        (news_df_grouped.index <= row['datetime_5'])]
            
            # If there's at least one event in the window, mark the new column as 1
            if not events_in_window.empty:
                self.df.at[index, 'news_event_5'] = 1

                # add a column for the width of bollinger bands
        self.df.loc[:, 'bollinger_width'] = finta.TA.BBWIDTH(self.df, period=20)
        # add a column for the awesome oscillator
        self.df.loc[:, 'awesome_oscillator'] = finta.TA.AO(self.df)


        # calculates the difference between consecutive elements in the prices array using the np.diff function. 
        # This results in an array that is one element shorter than the original prices array. To compensate for this, 
        # a zero is inserted at the beginning of the diff array using np.insert.
        # diff = np.insert(np.diff(prices), 0, 0)

        features = ['open', 'high', 'low', 'close', 'sma', 'smoothing_sma',
       'sma_diff', 'sma_sign', 'sma_crossover', 
       'bollinger_width', 'awesome_oscillator', 'day_of_week_transition',
       'news_event_5', 'secs_until_next_news_event']

        # drop rows with null values in features
        self.df = self.df.dropna(subset=features)
        
        signal_features = self.df.loc[:, features].to_numpy()[start:end]

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0  # pip

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]

            if self.spread_bool:
                if self._position == Positions.Short:
                    last_trade_price = self.prices[self._last_trade_tick] - self.spread
                    price_diff = current_price - last_trade_price
                    step_reward += -price_diff * 10000
                elif self._position == Positions.Long:
                    last_trade_price = self.prices[self._last_trade_tick] + self.spread
                    price_diff = current_price - last_trade_price
                    step_reward += price_diff * 10000
            else:
                last_trade_price = self.prices[self._last_trade_tick]
                price_diff = current_price - last_trade_price
                if self._position == Positions.Short:
                    step_reward += -price_diff * 10000
                elif self._position == Positions.Long:
                    step_reward += price_diff * 10000


        return step_reward

    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
          
            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    quantity = self._total_profit * (last_trade_price - self.trade_fee)
                    self._total_profit = quantity / current_price

            elif self.unit_side == 'right':
                if self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price # quantity is to get the number of lots
                    self._total_profit = quantity * (current_price - self.trade_fee)

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit
