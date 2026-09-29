@api
Feature: Console sessions and accounts
  Console access tokens are short-lived; refresh tokens rotate, and account
  changes sign out every session the user already holds.

  Scenario: A refresh token can be used only once
    Given a new console admin is signed in
    When they renew their session
    Then the renewed access token works
    When they try to renew with the same refresh token again
    Then the response status is 401

  Scenario: Changing a password signs out every session
    Given a new console admin is signed in
    When they change their password to "a-brand-new-password"
    Then their earlier access token is rejected
    And they can sign in with the new password but not the old one

  Scenario: A wrong current password is refused
    Given a new console admin is signed in
    When they try to change their password with a wrong current password
    Then the response status is 400
    And the error detail mentions "Current password is incorrect"

  Scenario: Admins cannot lock themselves out
    Given a new console admin is signed in
    When they try to demote themselves to viewer
    Then the response status is 409
    And the error detail mentions "cannot demote or deactivate your own"
    When they try to delete their own account
    Then the response status is 409
