using TensorBoardLogger
using Logging
using TensorBoardLogger: preprocess, summary_impl
using TensorBoardLogger: IntervalDomain, DiscreteDomain, HParam, Metric, HParamsConfig
using Test

test_log_dir = "test_logs/"
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@testset "HParamConfig Logger" begin

    # Declare the hyperparameter experiment in the root folder
    logger = TBLogger(test_log_dir*"hparamconfig", tb_overwrite)
    step = 1

    interval_domain = IntervalDomain(0.1, 3.0)
    hparam1 = HParam("interval_hparam", interval_domain, "display_name1", "description1")

    discrete_domain_strs = ["a", "b", "c"]
    discrete_domain = DiscreteDomain(discrete_domain_strs)
    hparam2 = HParam("discrete_domain_hparam", discrete_domain, "display_name2", "description2")

    hparams = [hparam1, hparam2]
    metric_name = "score"
    metric_display_name = "Score"
    metric = Metric(metric_name, "group", metric_display_name, "description", :DATASET_VALIDATION)
    metrics = [metric]
    hparams_config = HParamsConfig(hparams, metrics, 1.2)
    ss = TensorBoardLogger.hparams_config_summary(hparams_config)

    @test isa(ss, TensorBoardLogger.Summary_Value)
    @test ss.tag == TensorBoardLogger.EXPERIMENT_TAG

    # TODO: Deserialize and test more properties

    @test π != log_hparams_config(logger, hparams_config ;step=step)

    close.(values(logger.all_files))

    # Initialise an experiment with some of the hyperparameters set
    logger_experiment = TBLogger(test_log_dir*"hparamconfig/run", tb_overwrite)
    hparams_dict = Dict(
        hparam1=>0.5,
        hparam2=>"b"
    )
    @test π != log_hparams(logger_experiment, hparams_dict, "group_name", "trial_id", nothing ;step=step)


    @test π != log_value(logger_experiment, metric_name, 1.0, step=1)
    @test π != log_value(logger_experiment, metric_name, 10.0, step=2)
    

    close.(values(logger_experiment.all_files))
end

@testset "HParams Logger" begin
    logger = TBLogger(test_log_dir*"hparams", tb_overwrite)
    step = 1

    interval_domain = IntervalDomain(0.1, 3.0)
    hparam1 = HParam("interval_hparam", interval_domain, "display_name1", "description1")

    discrete_domain_strs = ["a", "b", "c"]
    discrete_domain = DiscreteDomain(discrete_domain_strs)
    hparam2 = HParam("discrete_domain_hparam", discrete_domain, "display_name2", "description2")

    hparams_dict = Dict(hparam1 => 1.2, hparam2 => "b")

    ss = TensorBoardLogger.hparams_summary(hparams_dict, "group_name", "trial_id", nothing)

    @test isa(ss, TensorBoardLogger.Summary_Value)
    @test ss.tag == TensorBoardLogger.SESSION_START_INFO_TAG

    # TODO: Deserialize and test more properties
    @test π != log_hparams(logger, hparams_dict, "group_name", "trial_id", nothing ;step=step)
    close.(values(logger.all_files))
end
