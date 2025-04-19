
def allcause(info):
    import joblib
    clf = joblib.load('best_xgb_allcause.pkl')

    death_proba = clf.predict_proba(info)[0][0]

    return death_proba


def card(info):
    import joblib
    clf = joblib.load('best_xgb_card.pkl')

    death_proba = clf.predict_proba(info)[0][0]

    return death_proba
