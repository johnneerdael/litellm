import React, { useState, useEffect } from "react";
import { Button, Alert, Card, List, Typography, Spin, Modal } from "antd";
import { UserOutlined, DeleteOutlined, ReloadOutlined, LoginOutlined } from "@ant-design/icons";

const { Text, Title } = Typography;

interface AntigravityAccount {
  email: string;
  project_id?: string;
  is_rate_limited: boolean;
  is_invalid: boolean;
}

interface AntigravityAccountsResponse {
  total: number;
  available: number;
  rate_limited: number;
  invalid: number;
  accounts: AntigravityAccount[];
}

interface AntigravityOAuthProps {
  accessToken: string;
}

const AntigravityOAuth: React.FC<AntigravityOAuthProps> = ({ accessToken }) => {
  const [accounts, setAccounts] = useState<AntigravityAccount[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ total: 0, available: 0, rate_limited: 0, invalid: 0 });

  const fetchAccounts = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/antigravity/accounts", {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch accounts: ${response.statusText}`);
      }
      const data: AntigravityAccountsResponse = await response.json();
      setAccounts(data.accounts);
      setStats({
        total: data.total,
        available: data.available,
        rate_limited: data.rate_limited,
        invalid: data.invalid,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAccounts();
  }, [accessToken]);

  const handleAddAccount = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/antigravity/auth/start", {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to start OAuth: ${response.statusText}`);
      }
      const data = await response.json();
      window.open(data.auth_url, "_blank", "width=600,height=700");
      
      Modal.info({
        title: "Complete Authentication",
        content: (
          <div>
            <p>A new window has opened for Google authentication.</p>
            <p>After completing authentication, click the button below to refresh the account list.</p>
          </div>
        ),
        okText: "Refresh Accounts",
        onOk: () => fetchAccounts(),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAccount = async (email: string) => {
    Modal.confirm({
      title: "Delete Account",
      content: `Are you sure you want to remove ${email}?`,
      okText: "Delete",
      okType: "danger",
      onOk: async () => {
        try {
          const response = await fetch(`/antigravity/accounts/${encodeURIComponent(email)}`, {
            method: "DELETE",
            headers: {
              Authorization: `Bearer ${accessToken}`,
            },
          });
          if (!response.ok) {
            throw new Error(`Failed to delete account: ${response.statusText}`);
          }
          fetchAccounts();
        } catch (err) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      },
    });
  };

  const handleResetRateLimits = async () => {
    try {
      const response = await fetch("/antigravity/accounts/reset-rate-limits", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to reset rate limits: ${response.statusText}`);
      }
      fetchAccounts();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  return (
    <Card title="Antigravity Accounts" style={{ marginBottom: 16 }}>
      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}

      <div style={{ marginBottom: 16, display: "flex", gap: 8 }}>
        <Button type="primary" icon={<LoginOutlined />} onClick={handleAddAccount} loading={loading}>
          Add Google Account
        </Button>
        <Button icon={<ReloadOutlined />} onClick={fetchAccounts} loading={loading}>
          Refresh
        </Button>
        {stats.rate_limited > 0 && (
          <Button icon={<ReloadOutlined />} onClick={handleResetRateLimits}>
            Reset Rate Limits
          </Button>
        )}
      </div>

      <div style={{ marginBottom: 16 }}>
        <Text>
          Total: {stats.total} | Available: {stats.available} | Rate Limited: {stats.rate_limited} | Invalid:{" "}
          {stats.invalid}
        </Text>
      </div>

      {loading ? (
        <Spin />
      ) : accounts.length === 0 ? (
        <Alert
          message="No accounts configured"
          description="Add a Google account to use Antigravity models (Claude & Gemini via Google Cloud Code)."
          type="info"
          showIcon
        />
      ) : (
        <List
          size="small"
          dataSource={accounts}
          renderItem={(account) => (
            <List.Item
              actions={[
                <Button
                  key="delete"
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => handleDeleteAccount(account.email)}
                />,
              ]}
            >
              <List.Item.Meta
                avatar={<UserOutlined />}
                title={account.email}
                description={
                  <>
                    {account.project_id && <Text type="secondary">Project: {account.project_id}</Text>}
                    {account.is_rate_limited && (
                      <Text type="warning" style={{ marginLeft: 8 }}>
                        (Rate Limited)
                      </Text>
                    )}
                    {account.is_invalid && (
                      <Text type="danger" style={{ marginLeft: 8 }}>
                        (Invalid)
                      </Text>
                    )}
                  </>
                }
              />
            </List.Item>
          )}
        />
      )}
    </Card>
  );
};

export default AntigravityOAuth;
