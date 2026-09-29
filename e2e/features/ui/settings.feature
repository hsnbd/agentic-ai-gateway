@ui
Feature: Settings and console users

  Scenario: An admin adds a console user
    Given I am signed in to the console as the admin
    And I go to "/settings"
    When I add a console user "<unique>@e2e.test" with role "viewer"
    Then the user "<same>@e2e.test" is listed
    And the new user can sign in

  Scenario: System status is reported
    Given I am signed in to the console as the admin
    When I go to "/settings"
    Then the settings show the database and Redis as "Connected"
    And the subsystem "rag" is "active"
    And the subsystem "mcp" is "active"
