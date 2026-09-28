"""Реестр реальных датасетов: ЕДИНСТВЕННОЕ место, где описан каждый датасет.

Его читают scripts/download_data.py (что и куда скачать) и scripts/run.py (--dataset имя).
Поля:
  uci_id    — ID в ucimlrepo (None, если файл кладётся вручную)
  file      — путь относительно data/ (diploma_fs/data или data/ в корне репозитория)
  target    — целевая колонка, ЯВНО (метаданные UCI бывают неверны: у 851 и 560 там не та цель)
  time_col  — столбец времени или None; если задан, разбиение по времени (только прошлое -> будущее)
  time_format — формат даты для pd.to_datetime (ОБЯЗАТЕЛЬНО для dd/mm: иначе 01/02 читается
              как 2 января и порядок ряда ломается молча); None — ISO/авто
  drop      — колонки-утечки и идентификаторы (удаляются до всего остального)
  cat_cols  — категориальные, даже если записаны числами (час, сезон, станция)
  sample    — сколько строк брать (None = все); для временных берётся непрерывный хвост ряда
  role      — main | classic | control (отрицательный контроль: ожидаем, что отбор не поможет)
  status    — draft (не проверен на скачанном файле) | verified
Правило отбора (зафиксировано до прогона, docs/DECISIONS.md): регрессия, >= 1000 строк,
5..40 числовых признаков, числовая цель задана явно, утечки удалены; spread — только описание.
"""

DATASETS = {
    "tetouan": dict(uci_id=849, file="849_tetouan.csv", target="Zone 1 Power Consumption",
                    time_col="DateTime", drop=["Zone 2  Power Consumption", "Zone 3  Power Consumption"],
                    cat_cols=[], sample=10000, role="main", status="draft"),
    "steel": dict(uci_id=851, file="851_steel.csv", target="Usage_kWh", time_col="date",
                  time_format="%d/%m/%Y %H:%M", drop=["CO2(tCO2)"], cat_cols=["WeekStatus", "Day_of_week", "Load_Type"],
                  sample=10000, role="main", status="draft"),
    "seoul_bike": dict(uci_id=560, file="560_seoul_bike.csv", target="Rented Bike Count",
                       time_col="Date", time_format="%d/%m/%Y", drop=[],
                       cat_cols=["Hour", "Seasons", "Holiday", "Functioning Day"],
                       sample=None, role="main", status="draft"),
    "temp_forecast": dict(uci_id=514, file="514_temp_forecast.csv", target="Next_Tmax",
                          time_col="Date", time_format="%Y-%m-%d", drop=["Next_Tmin"], cat_cols=["station"],
                          sample=None, role="main", status="draft"),
    "gas_turbine": dict(uci_id=551, file="551_gas_turbine.csv", target="NOX", time_col=None,
                        drop=["CO", "Year"], cat_cols=[], sample=10000, role="main",
                        status="draft"),
    "auction": dict(uci_id=713, file="713_auction.csv", target="verification.time",
                    time_col=None, drop=["verification.result"], cat_cols=[],
                    sample=None, role="main", status="draft"),
    "thermography": dict(uci_id=925, file="925_thermography.csv", target="aveOralM",
                         time_col=None, drop=["aveOralF"], cat_cols=["Gender", "Age", "Ethnicity"],
                         sample=None, role="main", status="draft"),
    "lattice": dict(uci_id=1091, file="1091_lattice.csv", target="k-inf", time_col=None,
                    drop=["PPPF"], cat_cols=[], sample=5000, role="main", status="draft"),
    "concrete": dict(uci_id=165, file="165_concrete.csv",
                     target="Concrete compressive strength", time_col=None, drop=[],
                     cat_cols=[], sample=None, role="classic", status="draft"),
    "ccpp": dict(uci_id=294, file="294_ccpp.csv", target="PE", time_col=None, drop=[],
                 cat_cols=[], sample=None, role="classic", status="draft"),
    "wine_quality": dict(uci_id=186, file="186_wine_quality.csv", target="quality",
                         time_col=None, drop=[], cat_cols=["color"], sample=None,
                         role="control", status="draft"),
    "garment": dict(uci_id=597, file="597_garment.csv", target="actual_productivity",
                    time_col="date", time_format="%m/%d/%Y", drop=[], cat_cols=["quarter", "department", "day", "team"],
                    sample=None, role="control", status="draft"),
}
# Имена колонок в draft-записях взяты из карточек UCI и будут сверены с реально скачанными
# файлами (у Tetouan в названиях зон бывают двойные пробелы, у Gas Turbine — NOX/NOx).
