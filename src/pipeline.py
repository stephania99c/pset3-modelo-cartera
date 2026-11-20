"""Construccion de features del modelo de priorizacion de cartera."""

import pandas as pd


def suma_ventana(obt, columna, desde, hasta):
    """Para cada credito y corte d, suma `columna` en los dias d+desde ... d+hasta."""
    eventos = obt.loc[obt[columna] > 0, ["numero_operacion", "fecha_corte", columna]]
    partes = []
    for k in range(desde, hasta + 1):
        partes.append(eventos.assign(fecha_corte=eventos["fecha_corte"] - pd.Timedelta(days=k)))
    total = pd.concat(partes).groupby(["numero_operacion", "fecha_corte"], as_index=False)[columna].sum()
    unido = obt[["numero_operacion", "fecha_corte"]].merge(total, on=["numero_operacion", "fecha_corte"], how="left")
    return unido[columna].fillna(0).to_numpy()


def agregar_features(obt):
    obt = obt.copy()
    obt["fecha_corte"] = pd.to_datetime(obt["fecha_corte"])
    obt["fecha_proxima_cuota"] = pd.to_datetime(obt["fecha_proxima_cuota"])

    # 1. Plazo terminado: todas las cuotas pendientes ya vencieron
    obt["plazo_terminado"] = ((obt["cuotas_por_pagar"] > 0)
                              & (obt["cuotas_por_pagar"] == obt["cuotas_mora"])).astype(int)

    # 2. Compromiso de pago registrado en la ultima semana (dias d-6 a d)
    obt["compromiso_semana_previa"] = (suma_ventana(obt, "compromiso_pago_dia", -6, 0) > 0).astype(int)

    # 3. La proxima cuota vence en los proximos 7 dias (dias d+1 a d+7)
    dias_a_cuota = (obt["fecha_proxima_cuota"] - obt["fecha_corte"]).dt.days
    obt["cuota_por_vencer_7d"] = dias_a_cuota.between(1, 7).astype(int)

    # Tratamiento para el efecto incremental: gestionado en los dias d+1 a d+7
    obt["gestionado_7d"] = (suma_ventana(obt, "num_gestiones_dia", 1, 7) > 0).astype(int)
    return obt