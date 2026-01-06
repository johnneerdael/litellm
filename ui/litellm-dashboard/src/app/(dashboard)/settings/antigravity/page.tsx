"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Button,
  Alert,
  Card,
  Table,
  Typography,
  Spin,
  Modal,
  Tag,
  Space,
  Statistic,
  Row,
  Col,
  Tooltip,
  Progress,
} from "antd";
import {
  UserOutlined,
  DeleteOutlined,
  ReloadOutlined,
  LoginOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const { Title, Text } = Typography;

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

const AntigravitySettingsPage = () => {
  const { accessToken } = useAuthorized();
  const [accounts, setAccounts] = useState<AntigravityAccount[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ total: 0, available: 0, rate_limited: 0, invalid: 0 });

  const fetchAccounts = useCallback(async () => {
    if (!accessToken) return;
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
  }, [accessToken]);

  useEffect(() => {
    fetchAccounts();
  }, [fetchAccounts]);

  const handleAddAccount = async () => {
    if (!accessToken) return;
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

  const columns = [
    {
      title: "Email",
      dataIndex: "email",
      key: "email",
      render: (email: string) => (
        <Space>
          <UserOutlined />
          <Text strong>{email}</Text>
        </Space>
      ),
    },
    {
      title: "Project ID",
      dataIndex: "project_id",
      key: "project_id",
      render: (projectId: string | undefined) => (
        <Text type="secondary">{projectId || "Auto-discovered"}</Text>
      ),
    },
    {
      title: "Status",
      key: "status",
      render: (_: unknown, record: AntigravityAccount) => {
        if (record.is_invalid) {
          return (
            <Tag icon={<ExclamationCircleOutlined />} color="error">
              Invalid
            </Tag>
          );
        }
        if (record.is_rate_limited) {
          return (
            <Tag icon={<ClockCircleOutlined />} color="warning">
              Rate Limited
            </Tag>
          );
        }
        return (
          <Tag icon={<CheckCircleOutlined />} color="success">
            Available
          </Tag>
        );
      },
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, record: AntigravityAccount) => (
        <Tooltip title="Remove account">
          <Button
            type="text"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteAccount(record.email)}
          />
        </Tooltip>
      ),
    },
  ];

  const availablePercent = stats.total > 0 ? Math.round((stats.available / stats.total) * 100) : 0;

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>Antigravity Accounts</Title>
      <Text type="secondary" style={{ marginBottom: 24, display: "block" }}>
        Manage Google accounts for accessing Claude and Gemini models via Antigravity (Google Cloud Code).
        Multiple accounts enable automatic failover when rate limits are hit.
      </Text>

      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="Total Accounts" value={stats.total} prefix={<UserOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Available"
              value={stats.available}
              valueStyle={{ color: "#3f8600" }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Rate Limited"
              value={stats.rate_limited}
              valueStyle={{ color: stats.rate_limited > 0 ? "#faad14" : undefined }}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Invalid"
              value={stats.invalid}
              valueStyle={{ color: stats.invalid > 0 ? "#cf1322" : undefined }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {stats.total > 0 && (
        <Card style={{ marginBottom: 24 }}>
          <Text>Account Health</Text>
          <Progress
            percent={availablePercent}
            status={availablePercent < 50 ? "exception" : availablePercent < 80 ? "normal" : "success"}
            format={() => `${stats.available}/${stats.total} available`}
          />
        </Card>
      )}

      <Card
        title="Configured Accounts"
        extra={
          <Space>
            <Button type="primary" icon={<LoginOutlined />} onClick={handleAddAccount} loading={loading}>
              Add Google Account
            </Button>
            <Button icon={<ReloadOutlined />} onClick={fetchAccounts} loading={loading}>
              Refresh
            </Button>
            {stats.rate_limited > 0 && (
              <Button onClick={handleResetRateLimits}>Reset Rate Limits</Button>
            )}
          </Space>
        }
      >
        {loading && accounts.length === 0 ? (
          <div style={{ textAlign: "center", padding: 40 }}>
            <Spin size="large" />
          </div>
        ) : accounts.length === 0 ? (
          <Alert
            message="No accounts configured"
            description={
              <div>
                <p>Add Google accounts to use Antigravity models (Claude & Gemini via Google Cloud Code).</p>
                <p>
                  <strong>Supported models:</strong> claude-sonnet-4.5-thinking, claude-opus-4.5-thinking,
                  claude-sonnet-4.5, gemini-3-flash, gemini-3-pro-high, gemini-3-pro-low, gemini-2.5-flash,
                  gemini-2.5-pro
                </p>
              </div>
            }
            type="info"
            showIcon
          />
        ) : (
          <Table
            dataSource={accounts}
            columns={columns}
            rowKey="email"
            pagination={accounts.length > 10 ? { pageSize: 10 } : false}
          />
        )}
      </Card>
    </div>
  );
};

export default AntigravitySettingsPage;
