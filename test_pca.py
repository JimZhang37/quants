from pca import fit_pca
import project_helper
import project_tests

num_factor_exposures = 20
pca = fit_pca(five_year_returns, num_factor_exposures, 'full')


project_tests.test_factor_betas(factor_betas)

risk_model = {}
risk_model['factor_betas'] = factor_betas(pca, five_year_returns.columns.values, np.arange(num_factor_exposures))

project_tests.test_factor_returns(factor_returns)

risk_model['factor_returns'] = factor_returns(
    pca,
    five_year_returns,
    five_year_returns.index,
    np.arange(num_factor_exposures))

project_tests.test_factor_cov_matrix(factor_cov_matrix)

ann_factor = 252
risk_model['factor_cov_matrix'] = factor_cov_matrix(risk_model['factor_returns'], ann_factor)

project_tests.test_idiosyncratic_var_matrix(idiosyncratic_var_matrix)

risk_model['idiosyncratic_var_matrix'] = idiosyncratic_var_matrix(five_year_returns, risk_model['factor_returns'], risk_model['factor_betas'], ann_factor)

project_tests.test_idiosyncratic_var_vector(idiosyncratic_var_vector)

risk_model['idiosyncratic_var_vector'] = idiosyncratic_var_vector(five_year_returns, risk_model['idiosyncratic_var_matrix'])


project_tests.test_predict_portfolio_risk(predict_portfolio_risk)

all_weights = pd.DataFrame(np.repeat(1/len(universe_tickers), len(universe_tickers)), universe_tickers)

