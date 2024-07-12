# actions.py
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher

class ActionGreet(Action):

    def name(self) -> Text:
        return "action_greet"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_greet'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionGoodbye(Action):

    def name(self) -> Text:
        return "action_goodbye"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_goodbye'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionChatbotChallenge(Action):

    def name(self) -> Text:
        return "action_chatbot_challenge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_chatbot_challenge'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionInformationOnPytorch(Action):

    def name(self) -> Text:
        return "action_information_on_pytorch"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_information_on_pytorch'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionFeaturesOfPytorch(Action):
    def name(self) -> Text:
        return "action_features_of_pytorch"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_action_features_of_pytorch'][0]['text']
        dispatcher.utter_message(text=response)
        return []


class ActionImplementingFeaturesOrFixingBugs(Action):

    def name(self) -> Text:
        return "action_implementing_features_or_fixing_bugs"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_implementing_features_or_fixing_bugs'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAddingTutorials(Action):

    def name(self) -> Text:
        return "action_adding_tutorials"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_adding_tutorials'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionSubmittingPullRequestsToFixOpenIssues(Action):

    def name(self) -> Text:
        return "action_submitting_pull_requests_to_fix_open_issues"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_submitting_pull_requests_to_fix_open_issues'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchScript(Action):

    def name(self) -> Text:
        return "action_torchscript"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torchscript'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionImprovingCodeReadability(Action):

    def name(self) -> Text:
        return "action_improving_code_readability"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_improving_code_readability'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionContributionProcess(Action):

    def name(self) -> Text:
        return "action_contribution_process"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_contribution_process'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionParticipatingInOnlineDiscussions(Action):

    def name(self) -> Text:
        return "action_participating_in_online_discussions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_participating_in_online_discussions'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionReviewingOpenPullRequests(Action):

    def name(self) -> Text:
        return "action_reviewing_open_pull_requests"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_reviewing_open_pull_requests'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionImprovingDocumentationAndTutorials(Action):

    def name(self) -> Text:
        return "action_improving_documentation_and_tutorials"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_improving_documentation_and_tutorials'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAddingTestCasesToMakeCodebaseMoreRobust(Action):

    def name(self) -> Text:
        return "action_adding_test_cases_to_make_codebase_more_robust"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_adding_test_cases_to_make_codebase_more_robust'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchMonitor(Action):

    def name(self) -> Text:
        return "action_torch_monitor"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_monitor'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFx(Action):

    def name(self) -> Text:
        return "action_torch_fx"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_fx'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionNamedTensors(Action):

    def name(self) -> Text:
        return "action_named_tensors"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_named_tensors'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionPromotingPytorch(Action):

    def name(self) -> Text:
        return "action_promoting_pytorch"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_promoting_pytorch'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTriagingIssues(Action):

    def name(self) -> Text:
        return "action_triaging_issues"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_triaging_issues'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionOpenSourceDev(Action):

    def name(self) -> Text:
        return "action_open_source_dev"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_open_source_dev'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAddNewMaintainer(Action):

    def name(self) -> Text:
        return "action_add_new_maintainer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_add_new_maintainer'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchSignal(Action):

    def name(self) -> Text:
        return "action_torch_signal"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_signal'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchOverrides(Action):

    def name(self) -> Text:
        return "action_torch_overrides"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_overrides'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchSpecial(Action):

    def name(self) -> Text:
        return "action_torch_special"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_special'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchLinalg(Action):

    def name(self) -> Text:
        return "action_torch_linalg"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_linalg'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchHub(Action):

    def name(self) -> Text:
        return "action_torch_hub"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_hub'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFutures(Action):

    def name(self) -> Text:
        return "action_torch_futures"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_futures'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFunc(Action):

    def name(self) -> Text:
        return "action_torch_func"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_func'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFtt(Action):

    def name(self) -> Text:
        return "action_torch_fft"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_fft'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchCompiler(Action):

    def name(self) -> Text:
        return "action_torch_compiler"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_compiler'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchEnvVars(Action):

    def name(self) -> Text:
        return "action_torch_env_vars"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_env_vars'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchLogging(Action):

    def name(self) -> Text:
        return "action_torch_logging"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_logging'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchTensorboard(Action):

    def name(self) -> Text:
        return "action_torch_tensorboard"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_tensorboard'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchXLA(Action):

    def name(self) -> Text:
        return "action_torch_xla"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_xla'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchDistributions(Action):

    def name(self) -> Text:
        return "action_torch_distributions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_distributions'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchModelZoo(Action):

    def name(self) -> Text:
        return "action_torch_model_zoo"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_model_zoo'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFXSymbolicShapes(Action):

    def name(self) -> Text:
        return "action_torch_fx_symbolic_shapes"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_fx_symbolic_shapes'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFXInterpreter(Action):

    def name(self) -> Text:
        return "action_torch_fx_interpreter"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_fx_interpreter'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFXGraphManipulation(Action):

    def name(self) -> Text:
        return "action_torch_fx_graph_manipulation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_fx_graph_manipulation'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFXDebugging(Action):

    def name(self) -> Text:
        return "action_torch_fx_debugging"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_fx_debugging'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchFXLimitations(Action):

    def name(self) -> Text:
        return "action_torch_fx_limitations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_fx_limitations'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchConfig(Action):

    def name(self) -> Text:
        return "action_torch_config"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_config'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchTypeInfo(Action):

    def name(self) -> Text:
        return "action_torch_type_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_type_info'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchIO(Action):

    def name(self) -> Text:
        return "action_torch_io"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_io'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchProfiler(Action):

    def name(self) -> Text:
        return "action_torch_profiler"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_profiler'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchAutograd(Action):

    def name(self) -> Text:
        return "action_torch_autograd"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_autograd'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchData(Action):

    def name(self) -> Text:
        return "action_torch_data"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_data'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchMultiprocessing(Action):

    def name(self) -> Text:
        return "action_torch_multiprocessing"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_multiprocessing'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchJIT(Action):

    def name(self) -> Text:
        return "action_torch_jit"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_jit'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchOptim(Action):

    def name(self) -> Text:
        return "action_torch_optim"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_optim'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchNN(Action):

    def name(self) -> Text:
        return "action_torch_nn"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_nn'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchQuantization(Action):

    def name(self) -> Text:
        return "action_torch_quantization"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_quantization'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchRPC(Action):

    def name(self) -> Text:
        return "action_torch_rpc"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_rpc'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchTesting(Action):

    def name(self) -> Text:
        return "action_torch_testing"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_testing'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchVision(Action):

    def name(self) -> Text:
        return "action_torch_vision"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_vision'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchAudio(Action):

    def name(self) -> Text:
        return "action_torch_audio"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_audio'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchText(Action):

    def name(self) -> Text:
        return "action_torch_text"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_text'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionTorchSparse(Action):

    def name(self) -> Text:
        return "action_torch_sparse"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_torch_sparse'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskInstallation(Action):

    def name(self) -> Text:
        return "action_ask_installation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_installation'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskExamples(Action):

    def name(self) -> Text:
        return "action_ask_examples"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_examples'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskLearningResources(Action):

    def name(self) -> Text:
        return "action_ask_learning_resources"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_learning_resources'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskCommunity(Action):

    def name(self) -> Text:
        return "action_ask_community"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_community'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskVersions(Action):

    def name(self) -> Text:
        return "action_ask_versions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_versions'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskModels(Action):

    def name(self) -> Text:
        return "action_ask_models"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_models'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskPerformanceTips(Action):

    def name(self) -> Text:
        return "action_ask_performance_tips"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_performance_tips'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskCustomDatasets(Action):

    def name(self) -> Text:
        return "action_ask_custom_datasets"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_custom_datasets'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskFavoriteFeature(Action):

    def name(self) -> Text:
        return "action_ask_favorite_feature"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_favorite_feature'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskAboutDay(Action):

    def name(self) -> Text:
        return "action_ask_about_day"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_about_day'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskHobbies(Action):

    def name(self) -> Text:
        return "action_ask_hobbies"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_hobbies'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskForJoke(Action):

    def name(self) -> Text:
        return "action_ask_for_joke"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_for_joke'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskFavoriteLibrary(Action):

    def name(self) -> Text:
        return "action_ask_favorite_library"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_favorite_library'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionAskAboutPython(Action):

    def name(self) -> Text:
        return "action_ask_about_py"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_about_py'][0]['text']
        dispatcher.utter_message(text=response)
        return []

class ActionChatbotIntelligence(Action):

    def name(self) -> Text:
        return "action_chatbot_intelligence"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_chatbot_intelligence'][0]['text']
        dispatcher.utter_message(text=response)
        return []


class ActionAskTorchDeploy(Action):

    def name(self) -> Text:
        return "action_ask__torch_deploy"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = domain['responses']['utter_ask_torch_deploy'][0]['text']
        dispatcher.utter_message(text=response)
        return []



# class ActionDefaultFallback(Action):
#
#     def name(self) -> Text:
#         return "action_default_fallback"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(response="utter_default_fallback")
#
#         return [UserUtteranceReverted()]

class ActionTwoStageFallback(Action):

    def name(self) -> Text:
        return "action_two_stage_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(response="utter_ask_rephrase")

        return [UserUtteranceReverted()]
