@ui
Feature: Console sign-in
  Operators sign in to the console with their console account.

  Scenario: An admin signs in and lands on the dashboard
    Given I open the console sign-in page
    When I sign in as the admin
    Then I see the "Dashboard" page

  Scenario: A wrong password is rejected
    Given I open the console sign-in page
    When I sign in with email "admin@e2e.test" and password "definitely-wrong"
    Then I see a sign-in error
    And I am still on the sign-in page

  Scenario: A deep link survives signing in
    When I open the console at "/logs" without signing in
    Then I am on the sign-in page
    When I sign in as the admin
    Then I see the "Request logs" page

  Scenario: Signing out ends the session
    Given I am signed in to the console as the admin
    When I sign out
    Then I am on the sign-in page
    When I open the console at "/keys" without signing in
    Then I am on the sign-in page

  Scenario: An expired access token is renewed silently
    Given I am signed in to the console as the admin
    When my console access token stops being valid
    And I go to "/logs"
    Then I see the "Request logs" page

  Scenario: Signing out revokes the token on the server
    Given I am signed in to the console as the admin
    And I remember my console access token
    When I sign out
    Then the remembered console token is rejected by the gateway
