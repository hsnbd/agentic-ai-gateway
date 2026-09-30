@ui
Feature: Playground

  Scenario: Chat with a model through the gateway
    Given I am signed in to the console as the admin
    And I go to "/playground"
    When I choose the model "eval-chat"
    And I send the playground message "Hello from the playground"
    Then the playground shows an assistant reply containing "Reply to: Hello from the playground"

  Scenario: Follow-up turns send the whole conversation
    Given I am signed in to the console as the admin
    And I go to "/playground"
    When I choose the model "eval-anthropic"
    And I send the playground message "First question"
    Then the playground shows an assistant reply containing "Reply to: First question"
    When I send the playground message "Second question"
    Then the playground shows an assistant reply containing "Reply to: Second question"
