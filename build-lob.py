import pandas as pd
import numpy as np
from struct import unpack
from collections import namedtuple, Counter
from datetime import timedelta
from time import time
import time as t_sleep
import datetime as dt
import q_table


ITCH_STORE = 'datalibrary/data/itch.h5'
ORDER_BOOK_STORE = 'order_book.h5'
STOCK = 'AAPL'

with pd.HDFStore(ITCH_STORE) as store:
    stocks = store['R'].loc[:, ['stock_locate', 'stock']]
    trades = store['P'].append(store['Q'].rename(columns={'cross_price': 'price'}), sort=False).merge(stocks)
trades['value'] = trades.shares.mul(trades.price)
trades['value_share'] = trades.value.div(trades.value.sum())
trade_summary = trades.groupby('stock').value_share.sum().sort_values(ascending=False)

order_dict = {-1: 'sell', 1: 'buy'}
#%%
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
    orders = pd.concat([data['A'], data['F']], sort=False, ignore_index=True).loc[:, order_cols]

    for m in messages[2: -3]:
        data[m] = data[m].merge(orders, how='left')

    data['U'] = data['U'].merge(orders, how='left',
                                right_on='order_reference_number',
                                left_on='original_order_reference_number',
                                suffixes=['', '_replaced'])

    data['Q'].rename(columns={'cross_price': 'price'}, inplace=True)
    data['X']['shares'] = data['X']['cancelled_shares']
    data['X'] = data['X'].dropna(subset=['price'])

    data = pd.concat([data[m] for m in messages], ignore_index=True, sort=False)
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
                        ], sort=False).dropna(subset=['price']).fillna(0)
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

def sort_order_book(bid_side, ask_side):
    return

print("Before parsing messages")
messages = get_messages(date='07302019',stock=STOCK)



order_book = {-1: {}, 1: {}}
current_orders = {-1: Counter(), 1: Counter()}
message_counter = Counter()
nlevels = 100

start = time()
ticker = 93141044776.0 + 3600*1e6
limit_order_price = 0
bb_pos = 0
bb_size = 0


limit_order_book = dict()

AES = 50
def translate_state(oders, aes):
    return

n_units = 10
limit_buy_prices = []
units_placed = 0

for message in messages.itertuples():
    i = message[0]
    i_last_sale = 0
    if i % 1e5 == 0 and i > 0:
        print('{:,.0f}\t\t{}'.format(i, timedelta(seconds=time() - start)))
        #save_orders(STOCK, order_book, append=True)
        #order_book = {-1: {}, 1: {}}
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
        # t_sleep.sleep(3)
        # print("orderbook A: ", order_book)
    elif message.type in ['E', 'C', 'X', 'D', 'U']:

        # t_sleep.sleep(3)
        # print("orderbook ECX: ", order_book)
        if message.type == 'U':
            if not np.isnan(message.shares_replaced):
                price = int(message.price_replaced)
                shares = -int(message.shares_replaced)
        else:
            if not np.isnan(message.price):
                price = int(message.price)
                shares = -int(message.shares)



        if price is not None:
            current_orders[buysell].update({price: shares})
            if current_orders[buysell][price] <= 0:
                current_orders[buysell].pop(price)
            current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
            #order_book[buysell][message.timestamp] = new_order

            # adjust bb pos and bb size accordingly, or execute the order
            #print("limit_order_price, E=", limit_order_price, message.price)
            if price == limit_order_price and message.type != 'D':
                if (message.shares >= bb_pos - 1):
                    print("Execute our limit order at price", price)
                    print("Execute at time:", message.timestamp)
                    limit_buy_prices.append(price)
                    limit_order_price = 0
                    i_last_sale = i
                    # sorted(current_orders[-1].items()
                    if units_placed >= n_units:
                        break
                else:
                    bb_pos -= message.shares
                    bb_size -= message.shares
            elif limit_order_price !=0 and i - i_last_sale > 1e5 and i_last_sale != 0:
                limit_order_price = 0
                print("Replacing order to front of queue")
    else:
        continue

    current_time = float(dt.datetime.strftime(message.timestamp, '%H%M%S%f'))
    # print("timestamp", message.timestamp )
    if (current_time - ticker) >= 0.05:
        # print (len(current_orders[-1]))
        # print (len(current_orders[1]))
        # print (sorted(current_orders[-1].items()))
        # print (sorted(current_orders[1].items(), reverse=True))
        sorted_ask_side = sorted(current_orders[-1].items())
        sorted_bid_side = sorted(current_orders[1].items(), reverse=True)
        #if sorted_ask_side[0][0] > sorted_bid_side[0][0]:
        #    print('X')
        if limit_order_price == 0:
            print("init limit order price")
            ba_order = next(iter(sorted_ask_side))
            bb_order = next(iter(sorted_bid_side))
            limit_order_price = ba_order[0]
            i_last_sale = i 
            ##### Need to:
                # - Add in reference price
                
            ## Naive ref price, update as in LeHalle
            #ref_price = 0.5 * (ba_order[0] + bb_order[0])
            
            
            bb_pos = int(ba_order[1]/AES) + 1
            bb_size = int(ba_order[1]/AES) + 1

        # iterate items to find current limit order position
        for order in sorted_ask_side:
            if order[0] == limit_order_price:
                bb_size = int(order[1]/AES) + 1
                break

        #state_idx = int((bb_pos / bb_size) * 30)
        stay_score = q_table.h_Stay[min(bb_size, 29)][min(bb_pos, 29)]
        market_score = q_table.h_Mkt[min(bb_size, 29)][min(bb_pos, 29)]

        if i % 1e4 == 0:
            print("bb_pos, bb_size", bb_pos, bb_size  )

            print("mkt score:", market_score)
            print("stay score:", stay_score)

        if (market_score > stay_score):
            bb_order = next(iter(sorted_bid_side))
            print("Place market order at price: ", bb_order[0])
            break

        # t_sleep.sleep(3)
        ticker = current_time

