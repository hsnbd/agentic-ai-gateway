@ui
Feature: Request logs in the console

  Scenario: A gateway request can be found and inspected
    Given a chat request was sent through the gateway saying "find me in the logs <unique>"
    And I am signed in to the console as the admin
    When I go to "/logs"
    And I search the logs for that request
    Then the logs table shows that request
    When I open that request
    Then the request detail shows it was served by "openai/eval-chat#2"

  Scenario: Logs can be exported as CSV
    Given a chat request was sent through the gateway saying "export me <unique>"
    And I am signed in to the console as the admin
    When I go to "/logs"
    And I export the logs as CSV
    Then the downloaded CSV has a header row and that request
