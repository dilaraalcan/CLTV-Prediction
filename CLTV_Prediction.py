from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)




###########################
#Veri Seti Hikayesi
###########################
# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
# Bu şirketin ürün kataloğunda hediyelik eşyalar yer alıyor. Promosyon ürünleri olarak da düşünülebilir.
# Çoğu müşterisinin toptancı olduğu bilgisi de mevcut.



# Değişkenler

# InvoiceNo – Fatura Numarası
# Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.
# StockCode – Ürün kodu Her bir ürün için eşsiz numara.
# Description – Ürün ismi Quantity – Ürün adedi
# Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate – Fatura tarihi UnitPrice – Fatura fiyatı (Sterlin)
# CustomerID – Eşsiz müşteri numarası Country – Ülke ismi


##################################################################

# aykırı degerleri belirlemek için sınırları belirleme:
# alt limit
# üst limit
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# aykırı degerleri baskılama:
# aykırı degerleri alt limit ve üst limite eşitleme.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# veri seti okutma
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()




df = df[df["Country"] == "United Kingdom"]
df.dropna(inplace=True)

# iade'leri cıkartıyoruz.
df = df[~df["Invoice"].str.contains( "C" , na=False)]
df = df[df["Quantity"] > 0]
df.describe().T

# aykırı degerleri baskılıyoruz.
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# toplam tutar degerini hesaplıyoruz.
df["TotalPrice"] = df["Quantity"] * df["Price"]

# analizi yapmak istedigimiz gün için belirliyoruz.
today_date = dt.datetime(2011, 12, 11)


#lifetime
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (rfm'de analiz gününe göre, burada kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df.head()
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df["recency"] = cltv_df["recency"] / 7 #haftalık
cltv_df["T"] = cltv_df["T"] / 7 #haftalık
cltv_df = cltv_df[(cltv_df['frequency'] > 1)] # birden fazla alısveris

# BG/NBD Modelinin Kurulması
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df.head()

# GAMMA-GAMMA Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)
cltv_df.head()


# BG-NBD ve GG modeli ile CLTV hesaplama
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)


cltv.shape
cltv = cltv.reset_index()

cltv.head()

cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False)[10:30]


#1-50 arası Transform
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1, 50))
scaler.fit(cltv_final[["clv"]])
cltv_final["SCALED_CLTV"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="clv", ascending=False)[10:30]
cltv_final.head()


# 1. 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV degerleri:
########################################################################

cltv1 = ggf.customer_lifetime_value(bgf,
                                    cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'],
                                    cltv_df['monetary'],
                                    time=1,  # months
                                    freq="W",  # T haftalık
                                    discount_rate=0.01)

rfm_cltv1_final = cltv_df.merge(cltv1, on="Customer ID", how="left")

cltv12 = ggf.customer_lifetime_value(bgf,
                                     cltv_df['frequency'],
                                     cltv_df['recency'],
                                     cltv_df['T'],
                                     cltv_df['monetary'],
                                     time=12,  # months
                                     freq="W",  # T haftalık
                                     discount_rate=0.01)

rfm_cltv12_final = cltv_df.merge(cltv12, on="Customer ID", how="left")

rfm_cltv12_final.head()

rfm_cltv1_final.sort_values("clv", ascending=False).head(10)
rfm_cltv12_final.sort_values("clv", ascending=False).head(10)

# genel olarak kitle aynıdır. bu beklenendir.
#cltv bir örüntü olusturarak gittigini düşünürsek birinci ay ile on ikinci ay arasında değişim yaşanmaması olasıdır.
# on ikinci ayda değişiklik iki müşteri arasında değişiklik yaşanmasının sebebi, sıralaması yükselen müşterinin  6-12. aylar arasında yüklü alısveriş mikatarı bırakması olabilir.


# 1. 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyelim.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # months
                                   freq="W",  # T haftalık
                                   discount_rate=0.01)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.head()

scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_final[["clv"]])
cltv_final["SCALED_CLTV"] = scaler.transform(cltv_final[["clv"]])

cltv_final["cltv_segment"] = pd.qcut(cltv_final["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])
cltv_final["cltv_segment"].value_counts()
cltv_final.head()


cltv_final.groupby("cltv_segment")[["expected_purc_1_month", "expected_average_profit", "clv", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})


# B segmenti cltv degerini arttıracak kampanyalar sunulması durumunda A segmentine yükselebilir.

