@ui
Feature: Observability pages show live data
  The usage, cache, and guardrail pages are driven by real gateway traffic.

  Scenario: Usage lists a model that has served traffic
    Given I use the master key
    And I send a chat request for model "eval-chat" saying "usage page <unique>"
    And I am signed in to the console as the admin
    When I go to "/usage"
    Then I see the "Usage and costs" page
    When I group usage by "Model"
    Then the page shows "eval-chat"

  Scenario: The cache page reports that the cache is available
    Given I am signed in to the console as the admin
    When I go to "/cache"
    Then I see the "Semantic cache" page
    And the page shows "Available"

  Scenario: Guardrail policies are visible to viewers
    Given I am signed in to the console as a new viewer
    When I go to "/guardrails"
    Then I see the "Guardrails" page
    And the page shows "default"
    And the page shows no error
