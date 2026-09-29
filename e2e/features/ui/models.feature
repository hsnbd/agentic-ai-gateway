@ui
Feature: Models and routing in the console

  Scenario: Deployments show their routing configuration
    Given I am signed in to the console as the admin
    When I go to "/models"
    Then the deployment table shows "eval-router" with priority "20" and weight "9"
    And the deployment table shows tags "cheap, fast"

  Scenario: Running a deployment health check
    Given I am signed in to the console as the admin
    When I go to "/models"
    And I run a health check on the "eval-router" fallback chain
    Then I see a notification containing "healthy"
