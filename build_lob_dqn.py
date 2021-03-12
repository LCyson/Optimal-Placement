import pandas as pd
import numpy as np
from collections import namedtuple, Counter
from datetime import timedelta
from time import time
import time as t_sleep
import datetime as dt
import matplotlib.pyplot as plt
import keras

ITCH_STORE = 'datalibrary/data/itch.h5'
ORDER_BOOK_STORE = 'order_book.h5'
STOCK = 'TSLA'

with pd.HDFStore(ITCH_STORE) as store:
    stocks = store['R'].loc[:, ['stock_locate', 'stock']]
    trades = store['P'].append(store['Q'].rename(columns={'cross_price': 'price'})).merge(stocks)
trades['value'] = trades.shares.mul(trades.price)
trades['value_share'] = trades.value.div(trades.value.sum())
trade_summary = trades.groupby('stock').value_share.sum().sort_values(ascending=False)

order_dict = {-1: 'sell', 1: 'buy'}


# %%
def get_messages(date, stock):
    """Collect trading messages for given stock"""
    with pd.HDFStore(ITCH_STORE) as store:
        stock_locate = store.select('R', where='stock = stock').stock_locate.iloc[0]
        target = 'stock_locate = stock_locate'

        data = {}
        # trading message types
        messages = ['A', 'F', 'E', 'C', 'X', 'D', 'U', 'P', 'Q']
        for m in messages:
            data[m] = store.select(m, where=target).drop('stock_locate', axis=1).assign(type=m)

    order_cols = ['order_reference_number', 'buy_sell_indicator', 'shares', 'price']
    orders = pd.concat([data['A'], data['F']], ignore_index=True).loc[:, order_cols]

    for m in messages[2: -3]:
        data[m] = data[m].merge(orders, how='left')

    data['U'] = data['U'].merge(orders, how='left',
                                right_on='order_reference_number',
                                left_on='original_order_reference_number',
                                suffixes=['', '_replaced'])

    data['Q'].rename(columns={'cross_price': 'price'}, inplace=True)
    data['X']['shares'] = data['X']['cancelled_shares']
    data['X'] = data['X'].dropna(subset=['price'])

    data = pd.concat([data[m] for m in messages], ignore_index=True)
    data['date'] = pd.to_datetime(date, format='%m%d%Y')
    data.timestamp = data['date'].add(data.timestamp)
    data = data[data.printable != 0]

    drop_cols = ['tracking_number', 'order_reference_number', 'original_order_reference_number',
                 'cross_type', 'new_order_reference_number', 'attribution', 'match_number',
                 'printable', 'date', 'cancelled_shares']
    return data.drop(drop_cols, axis=1).sort_values('timestamp').reset_index(drop=True)


def get_trades(m):
    """Combine C, E, P and Q messages into trading records"""
    trade_dict = {'executed_shares': 'shares', 'execution_price': 'price'}
    cols = ['timestamp', 'executed_shares']
    trades = pd.concat([m.loc[m.type == 'E', cols + ['price']].rename(columns=trade_dict),
                        m.loc[m.type == 'C', cols + ['execution_price']].rename(columns=trade_dict),
                        m.loc[m.type == 'P', ['timestamp', 'price', 'shares']],
                        m.loc[m.type == 'Q', ['timestamp', 'price', 'shares']].assign(cross=1),
                        ]).dropna(subset=['price']).fillna(0)
    return trades.set_index('timestamp').sort_index().astype(int)


def add_orders(orders, buysell, nlevels):
    """Add orders up to desired depth given by nlevels;
        sell in ascending, buy in descending order
    """
    new_order = []
    items = sorted(orders.copy().items())
    if buysell == 1:
        items = reversed(items)
    for i, (p, s) in enumerate(items, 1):
        new_order.append((p, s))
        if i == nlevels:
            break
    return orders, new_order


def save_orders(stock, orders, append=False):
    cols = ['price', 'shares']
    for buysell, book in orders.items():
        df = (pd.concat([pd.DataFrame(data=data,
                                      columns=cols)
                        .assign(timestamp=t)
                         for t, data in book.items()]))
        key = '{}/{}'.format(stock, order_dict[buysell])
        df.loc[:, ['price', 'shares']] = df.loc[:, ['price', 'shares']].astype(int)
        with pd.HDFStore(ORDER_BOOK_STORE) as store:
            if append:
                store.append(key, df.set_index('timestamp'), format='t')
            else:
                store.put(key, df.set_index('timestamp'))


def generate_state(sorted_ask_side, sorted_bid_side, tick_idx, queue_idx, time_left, share_left):
    bid_side_queue = [max(int(number / AES), 1) for _, number in sorted_bid_side]
    ask_side_queue = [max(int(number / AES), 1) for _, number in sorted_ask_side]
    normalized_time_left = int(100 * time_left / (20 * 1e6))

    new_state = np.concatenate(
        (bid_side_queue[-3:], ask_side_queue[-3:], [tick_idx], [queue_idx], [normalized_time_left], [share_left]))

    print("generating new state", new_state)
    return new_state


def find_current_position(limit_order_price, sorted_bid_side):
    new_tick_pos = -1

    for idx, order in enumerate(reversed(sorted_bid_side)):
        if order[0] == limit_order_price:
            new_tick_pos = idx
            break

    assert (new_tick_pos != -1)
    return new_tick_pos


order_book = {-1: {}, 1: {}}
current_orders = {-1: Counter(), 1: Counter()}
message_counter = Counter()
nlevels = 100

# define initial states
start = time()
placement_start_time = 40000023941.0 + 7200 * 1e6
limit_order_price = 0
AES = 20
tick_pos = 0
queue_pos = 0
time_left = 200 * 1e6
order_number = 1000

# init metric variables
n_units = 10
limit_buy_prices = []
buy_price_distances = []
order_placement_time = 0
optimal_order_price = 0

agent = keras.models.load_model("dqn-model")

messages = get_messages(date='07302019', stock=STOCK)
for message in messages.itertuples():
    i = message[0]
    if i % 1e5 == 0 and i > 0:
        print('{:,.0f}\t\t{}'.format(i, timedelta(seconds=time() - start)))
        # save_orders(STOCK, order_book, append=True)
        # order_book = {-1: {}, 1: {}}
        start = time()
    if np.isnan(message.buy_sell_indicator):
        continue
    message_counter.update(message.type)

    buysell = message.buy_sell_indicator
    price, shares = None, None

    if message.type in ['A', 'F', 'U']:
        price = int(message.price)
        shares = int(message.shares)

        current_orders[buysell].update({price: shares})
        current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
        order_book[buysell][message.timestamp] = new_order
    elif message.type in ['E', 'C', 'X', 'D', 'U']:
        if message.type == 'U':
            if not np.isnan(message.shares_replaced):
                price = int(message.price_replaced)
                shares = -int(message.shares_replaced)
        else:
            if not np.isnan(message.price):
                price = int(message.price)
                shares = -int(message.shares)

        if price is not None:
            if price == limit_order_price and (message.type == 'D' or message.type == 'U' or message.type == 'X'):
                print("Historical order is cancelled/replaced/deleted")
                limit_order_price = 0

            current_orders[buysell].update({price: shares})
            if current_orders[buysell][price] <= 0:
                current_orders[buysell].pop(price)
            current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)

            # record the lowest order price during a placement period
            if optimal_order_price > price and (message.type == 'E' or message.type == 'C'):
                print("optimal_order_price", "price", optimal_order_price, price)
                optimal_order_price = price

            # adjust bb pos and bb size accordingly, or execute the order
            if price == limit_order_price and message.type != 'D':
                # if (message.shares >= queue_pos - 1):
                print("Execute our limit order at price", price)
                print("Execute at time:", message.timestamp)
                limit_buy_prices.append(price)
                optimal_order_price = min(optimal_order_price, price)
                buy_price_distances.append(optimal_order_price - price)
                limit_order_price = 0
                order_placement_time = current_time
                order_number -= 1
                # else:
                #     queue_pos -= message.shares
    else:
        continue

    # start placing buy order(s)
    current_time = float(dt.datetime.strftime(message.timestamp, '%H%M%S%f'))
    if current_time - placement_start_time >= 0:
        time_left = max(0, time_left - (current_time - order_placement_time))
        sorted_ask_side = sorted(current_orders[-1].items(), reverse=True)
        sorted_bid_side = sorted(current_orders[1].items())

        # reset order state each time we execute or cancel an order
        if limit_order_price == 0:
            print("init limit order price")
            limit_order_price = sorted_bid_side[-1][0]
            order_placement_time = current_time
            optimal_order_price = limit_order_price
            time_left = 200 * 1e6
            tick_pos = 0
            queue_pos = np.random.randint(0, max(1, sorted_bid_side[-1][1] - 1))
        else:
            new_tick_pos = find_current_position(limit_order_price, sorted_bid_side)

            state = generate_state(sorted_ask_side, sorted_bid_side, new_tick_pos, queue_pos, time_left, 1)
            actions = agent.model.predict(np.array([state]))
            action = np.where(actions[0] == np.max(actions[0]))[0][0]
            print(sorted_ask_side)
            print(sorted_bid_side)
            print(actions)
            print(action)
            print(new_tick_pos)
            print(limit_order_price)

            # take action according to DQN action result
            #    - (0) Nothing, stay in queue
            #    - (1) Increase 1 tick, if at tick 1 - market buy
            #    - (2) Decrease 1 tick, if at last tick - stay in queue
            #    - (3) Market buy
            if action == 3 or (action == 1 and tick_pos == 0):
                market_order_price = sorted_ask_side[-1][0]
                print("Execute our market order at price", market_order_price)
                print("Execute at time:", float(dt.datetime.strftime(message.timestamp, '%H%M%S%f')))
                limit_buy_prices.append(market_order_price)
                optimal_order_price = min(optimal_order_price, price)
                buy_price_distances.append(optimal_order_price - market_order_price)
                limit_order_price = 0
                order_number -= 1
            elif action == 2:
                if tick_pos != 2:
                    limit_order_price = sorted_bid_side[-tick_pos - 2][0]
                    tick_pos += 1
                    queue_pos = np.random.randint(0, sorted_bid_side[-tick_pos-1][1] - 1)
            elif action == 1:
                limit_order_price = sorted_bid_side[-tick_pos + 1][0]
                tick_pos -= 1
                queue_pos = np.random.randint(0, sorted_bid_side[-tick_pos-1][1] - 1)


        # early exit if all orders have been executed
        if order_number == 0:
            print("Execute all buy orders")
            break

        t_sleep.sleep(1)

print("Finish simulating!")

np.save('limit_buy_prices_save.npy', limit_buy_prices)
np.save('buy_price_distances_save.npy', buy_price_distances)

print("Average of buy_price_distances is", np.mean(buy_price_distances))
print("Median of buy_price_distances is", np.median(buy_price_distances))
print("Standard Deviation of buy_price_distances is ", np.std(buy_price_distances))
